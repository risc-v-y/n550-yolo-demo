"""Version-2, two-slot UINT8 PCIe transport. One worker owns all tool calls.

Upload inactive input while the board consumes the active slot. Commit only
after downloading the preceding result. Neither a failed write nor an entire
frame is retried here. Status/result CRCs reject partial publication.
"""
import struct
import time
import zlib
from pcie_backend import PCIeBackend, PBTools, MAILBOX, MAGIC, READY, DONE, ERROR, checked, STATUS, STATUS_KEYS


class RequestReadbackError(ValueError):
    def __init__(self,details):
        self.details=details
        super().__init__('Input request readback mismatch: frame_id={} slot={} active_frame_id={}; '
                         'request was not committed'.format(details['frame_id'],details['slot'],
                                                           details['active_frame_id']))


class StreamBackend(PCIeBackend):
    required_version=2

    def __init__(self,*args,profile=False,mode='diagnostic',**kwargs):
        if mode=='performance' and profile: raise ValueError('Performance firmware has no node profile')
        kwargs['mode']=mode
        super().__init__(*args,**kwargs)
        self.profile=profile
        self.options=0 if profile else 16
        self.pending=None
        self.active=None
        self.last_profile=None
        try:
            s=self.stream
            # Protocol-2 reserved capability word declares supported pack bits.
            # Zero is the original firmware's PRECONVERT|SPATIAL capability.
            s['pack_mask']=int(s.get('reserved',0)) or 3
            if (s['input_bytes'],s['slot_bytes'],s['result_bytes'])!=(519168,519232,7336):
                raise ValueError('Unsupported stream sizes; update host and firmware together')
            for address,size in ((s['slots'],2*s['slot_bytes']), (s['result'],s['result_bytes']),
                                 (s['profile'],24320)):
                PBTools._range(address,size)
                if address%64: raise ValueError('Unaligned stream address')
        except BaseException:
            self.close()
            raise

    def configure(self,config):
        if self.active or self.pending or self.failed:
            raise RuntimeError('Configuration requires a healthy idle session')
        m,n,pack,compute,graph=[int(config.get(k,d)) for k,d in
                               (('m',32),('n',32),('pack',0),('compute',0),('graph',0))]
        if (m not in (32,64,128) or n not in (32,64,128) or
                max(m,n)>self.stream['max_rows'] or m*n*4>self.stream['accumulator_bytes'] or
                min(pack,compute,graph)<0 or pack&~self.stream['pack_mask'] or
                pack&~31 or compute&~7 or graph&~15):
            raise ValueError('Configuration exceeds reported hardware capacity')
        deadline=time.monotonic()+self.infer_timeout
        self.sequence=(self.sequence+1)&0xffffffff or 1
        body=struct.pack('<QQ11I',self.session,self.frame,6,self.sequence,0,0,self.options,m,n,pack,compute,graph,0)
        body+=struct.pack('<I',zlib.crc32(body))
        try:
            self.link.write(MAILBOX+64,body,deadline)
            if self.link.read(MAILBOX+64,64,deadline)!=body: raise ValueError('Config readback mismatch')
            self.link.write(MAILBOX+128,struct.pack('<16I',self.sequence,*([0]*15)),deadline)
            self._wait(READY,deadline)
        except BaseException:
            self.failed=True
            raise

    def upload(self,data,frame):
        if self.failed or self.pending: raise RuntimeError('No free pending slot/session failed')
        if not isinstance(data,bytes) or len(data)!=519168 or frame!=self.frame+1:
            raise ValueError('Expected next sequential UINT8 CHW frame')
        slot=(frame-1)%2
        if self.active and slot==self.active['slot']: raise RuntimeError('Input slot still owned by board')
        started=time.monotonic()
        deadline=started+self.infer_timeout
        sequence=(self.sequence+1)&0xffffffff or 1
        body=struct.pack('<QQ11I',self.session,frame,5,sequence,len(data),0 if self.fast else zlib.crc32(data),self.options,*([0]*6))
        body+=struct.pack('<I',zlib.crc32(body))
        address=self.stream['slots']+slot*self.stream['slot_bytes']
        try:
            self.link.write(address,body+data,deadline)
            actual=body if self.fast else self.link.read(address,64,deadline)
            if actual!=body:
                active=self.active['frame'] if self.active else None
                details={'type':'request_readback_mismatch','stage':'input_header_readback',
                         'frame_id':frame-1,'transport_frame':frame,'slot':slot+1,
                         'active_frame_id':active-1 if active is not None else None,
                         'sequence':sequence,'cpu_address':address,'pcie_address':address-0x80000000,
                         'expected_hex':body.hex(),'actual_hex':actual.hex(),
                         'actual_bytes':len(actual),
                         'different_offsets':[i for i in range(max(len(body),len(actual)))
                             if i>=len(body) or i>=len(actual) or body[i]!=actual[i]],
                         'actual_crc_valid':len(actual)==64 and
                             struct.unpack_from('<I',actual,60)[0]==zlib.crc32(actual[:60])}
                self.diagnostic(details)
                raise RequestReadbackError(details)
            self.pending={'frame':frame,'sequence':sequence,'slot':slot,'frame_started':started,
                          'upload_seconds':time.monotonic()-started}
        except BaseException:
            self.failed=True
            raise

    def begin(self):
        if self.failed or self.active or not self.pending: raise RuntimeError('Frame cannot be committed')
        item=self.pending
        deadline=time.monotonic()+self.infer_timeout
        commit_started=time.monotonic()
        try:
            self.link.write(MAILBOX+128,struct.pack('<16I',item['sequence'],item['slot']+1,*([0]*14)),deadline)
        except BaseException:
            self.failed=True
            raise
        self.frame=item['frame']; self.sequence=item['sequence']
        self.active=item; self.pending=None
        item['commit_seconds']=time.monotonic()-commit_started
        item['started']=time.monotonic(); item['deadline']=deadline

    def _result_packet(self,item):
        """The result status line is published last; one read obtains DONE+data.
        An old packet is ignored, never accepted for the new committed frame.
        """
        bad=0
        while True:
            self._check(item['deadline'])
            tick=time.monotonic()
            packet=self.link.read(self.stream['result'],7336,item['deadline'])
            try:
                if len(packet)!=7336: raise ValueError('Result packet length mismatch')
                record=dict(zip(STATUS_KEYS,STATUS.unpack(checked(packet[:64]))))
            except ValueError:
                bad+=1
                if bad>=3: raise ValueError('Repeated incomplete result header')
                continue
            if record['magic']!=MAGIC or record['version']!=2 or record['state'] not in range(6):
                raise ValueError('Invalid result status; firmware may have restarted')
            self.last_status=record
            if record['state']==ERROR and (record['sequence']==self.sequence or record['error'] not in range(0x40,0x45)):
                self._snapshot(item['deadline'])
                raise RuntimeError('Board PCIe error: {}; diagnostic={}'.format(record,self.last_diagnostic))
            if record['sequence']==self.sequence:
                if (record['session'],record['frame'])!=(self.session,self.frame):
                    raise ValueError('PCIe result session/frame mismatch')
                if record['state']==DONE:
                    if not self.fast and struct.unpack_from('<Q',packet,7328)[0]!=zlib.crc32(packet[:7328]):
                        bad+=1
                        if bad>=3: raise ValueError('Repeated result packet CRC mismatch')
                        continue
                    return record,packet,tick
            bad=0
            # A slow vendor call already throttles polling; do not add 0.1s.
            self.stop.wait(max(0,min(self.poll_interval-(time.monotonic()-tick),item['deadline']-time.monotonic())))

    def finish(self):
        if self.failed or not self.active: raise RuntimeError('No active frame')
        item=self.active
        try:
            if self.capabilities&8:
                status,packet,download=self._result_packet(item)
            else:
                status=self._wait(DONE,item['deadline'])
                download=time.monotonic()
                packet=self.link.read(self.stream['result'],7336,item['deadline'])
            if not self.fast and struct.unpack_from('<Q',packet,7328)[0]!=zlib.crc32(packet[:7328]):
                raise ValueError('Result packet CRC mismatch')
            record=dict(zip(STATUS_KEYS,STATUS.unpack(checked(packet[:64]))))
            if record!=status: raise ValueError('Result packet does not match completed frame')
            raw=packet[128:7328]
            if record['output_bytes']!=7200 or (not self.fast and zlib.crc32(raw)!=record['output_crc']):
                raise ValueError('Result size/CRC mismatch')
            if not self.fast: self.validate_result(raw)
            cycles=dict(zip('model_cycles service_cycles input_crc_cycles output_cycles input_pack bias_dequant profiled configuration'.split(),struct.unpack_from('<8Q',packet,64)))
            if bool(cycles['profiled'])!=self.profile: raise ValueError('Profile option was not honored')
            self.last_profile=None
            profile_start=time.monotonic()
            if self.profile:
                from board_profile import decode_profile
                blob=self.link.read(self.stream['profile'],24320,item['deadline'])
                self.last_profile=decode_profile(blob,self.frame,self.session)
            self.last_timing={'upload_seconds':item['upload_seconds'],
                'commit_seconds':item['commit_seconds'],
                'run_roundtrip_seconds':download-item['started'],
                'download_seconds':profile_start-download,
                'profile_read_seconds':time.monotonic()-profile_start if self.profile else None,
                'cycles':cycles}
            if self.fast:
                self.last_timing={'frame_seconds':time.monotonic()-item['frame_started'],
                                 'cycles':None,'profile_read_seconds':None}
            elif self.capabilities&8:
                self.last_timing['result_wait_seconds']=profile_start-item['started']
                self.last_timing['completion_read_seconds']=profile_start-download
                self.last_timing['run_roundtrip_seconds']=None
                self.last_timing['download_seconds']=None
            self.active=None
            return raw
        except BaseException:
            self.failed=True
            raise

    def infer_u8(self,data):
        self.upload(data,self.frame+1); self.begin()
        return self.finish()
