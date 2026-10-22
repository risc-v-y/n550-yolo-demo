"""Version-2, two-slot UINT8 PCIe transport. One worker owns all tool calls.

Upload inactive input while the board consumes the active slot. Commit only
after downloading the preceding result. Neither a failed write nor an entire
frame is retried here. Status/result CRCs reject partial publication.
"""
import struct
import time
import zlib
from pcie_backend import PCIeBackend, PBTools, MAILBOX, READY, DONE, checked, STATUS, STATUS_KEYS


class StreamBackend(PCIeBackend):
    required_version=2

    def __init__(self,*args,profile=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.profile=profile
        self.options=0 if profile else 16
        self.pending=None
        self.active=None
        self.last_profile=None
        try:
            s=self.stream
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
                pack&~3 or compute&~7 or graph&~15):
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
        body=struct.pack('<QQ11I',self.session,frame,5,sequence,len(data),zlib.crc32(data),self.options,*([0]*6))
        body+=struct.pack('<I',zlib.crc32(body))
        address=self.stream['slots']+slot*self.stream['slot_bytes']
        try:
            self.link.write(address,body+data,deadline)
            if self.link.read(address,64,deadline)!=body:
                raise ValueError('Input request readback mismatch; frame not committed')
            self.pending={'frame':frame,'sequence':sequence,'slot':slot,
                          'upload_seconds':time.monotonic()-started}
        except BaseException:
            self.failed=True
            raise

    def begin(self):
        if self.failed or self.active or not self.pending: raise RuntimeError('Frame cannot be committed')
        item=self.pending
        deadline=time.monotonic()+self.infer_timeout
        try:
            self.link.write(MAILBOX+128,struct.pack('<16I',item['sequence'],item['slot']+1,*([0]*14)),deadline)
        except BaseException:
            self.failed=True
            raise
        self.frame=item['frame']; self.sequence=item['sequence']
        self.active=item; self.pending=None
        item['started']=time.monotonic(); item['deadline']=deadline

    def finish(self):
        if self.failed or not self.active: raise RuntimeError('No active frame')
        item=self.active
        try:
            status=self._wait(DONE,item['deadline'])
            download=time.monotonic()
            packet=self.link.read(self.stream['result'],7336,item['deadline'])
            if struct.unpack_from('<Q',packet,7328)[0]!=zlib.crc32(packet[:7328]):
                raise ValueError('Result packet CRC mismatch')
            record=dict(zip(STATUS_KEYS,STATUS.unpack(checked(packet[:64]))))
            if record!=status: raise ValueError('Result packet does not match completed frame')
            raw=packet[128:7328]
            if record['output_bytes']!=7200 or zlib.crc32(raw)!=record['output_crc']:
                raise ValueError('Result size/CRC mismatch')
            self.validate_result(raw)
            cycles=dict(zip('model_cycles service_cycles input_crc_cycles output_cycles input_pack bias_dequant profiled configuration'.split(),struct.unpack_from('<8Q',packet,64)))
            if bool(cycles['profiled'])!=self.profile: raise ValueError('Profile option was not honored')
            self.last_profile=None
            profile_start=time.monotonic()
            if self.profile:
                from board_profile import decode_profile
                blob=self.link.read(self.stream['profile'],24320,item['deadline'])
                self.last_profile=decode_profile(blob,self.frame,self.session)
            self.last_timing={'upload_seconds':item['upload_seconds'],
                'run_roundtrip_seconds':download-item['started'],
                'download_seconds':profile_start-download,
                'profile_read_seconds':time.monotonic()-profile_start if self.profile else None,
                'cycles':cycles}
            self.active=None
            return raw
        except BaseException:
            self.failed=True
            raise

    def infer_u8(self,data):
        self.upload(data,self.frame+1); self.begin()
        return self.finish()
