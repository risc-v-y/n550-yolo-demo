"""Decode CPU-cycle counters published with PCIe DONE; Python 3.8, stdlib only."""
import hashlib
import struct
import zlib

PROFILE_MAGIC = 0x3152465036325955
PROFILE_BYTES = 8152
PROFILE_V2_BYTES = 24320
PROFILE_LENGTHS = (PROFILE_BYTES, PROFILE_V2_BYTES)
PHASES = ('weight_prepare', 'input_im2col', 'input_pack', 'matrix_submit',
          'matrix_wait', 'bias_dequant')
WEIGHT_FIELDS = ('weight_pack_bytes', 'weight_pack_values', 'weight_pack_cycles',
                 'prepared_layers', 'runtime_weight_conversions')
FIELDS = ('magic version bytes frame nodes completed options state model_cycles '
          'graph_cycles input_check_cycles service_cycles input_crc_cycles output_cycles '
          'node_cycles session').split()


def decode_profile(raw, frame, session):
    if len(raw) not in PROFILE_LENGTHS:
        raise ValueError('Board timing length mismatch')
    checksum = struct.unpack_from('<Q', raw, len(raw)-8)[0]
    if checksum != zlib.crc32(raw[:-8]):
        raise ValueError('Board timing CRC mismatch')
    report = dict(zip(FIELDS, struct.unpack_from('<16Q', raw)))
    version = report['version']
    if (version not in (1, 2) or report['magic'] != PROFILE_MAGIC or
            report['bytes'] != (PROFILE_BYTES if version == 1 else PROFILE_V2_BYTES) or
            report['bytes'] != len(raw)):
        raise ValueError('Unsupported board timing layout')
    if (report['frame'], report['session']) != (frame, session):
        raise ValueError('Board timing frame/session mismatch')
    if (report['nodes'], report['completed'], report['state']) != (334, 334, 1):
        raise ValueError('Incomplete board timing')
    if report['options'] & ~14:
        raise ValueError('Unknown board timing options')
    nodes = []
    for i in range(334):
        stride = 24 if version == 1 else 72
        cache, compute, check = struct.unpack_from('<3Q', raw, 128+i*stride)
        nodes.append(dict(node=i, cache=cache, compute=compute, check=check,
                          total=cache+compute+check))
        if version == 2:
            phases = struct.unpack_from('<6Q', raw, 128+i*stride+24)
            if sum(phases) > compute:
                raise ValueError('Node phase cycles exceed compute cycles')
            nodes[-1]['phases'] = dict(zip(PHASES, phases))
    if version == 2:
        offset = 128+334*72
        phases = dict((name, dict(zip(('ticks', 'calls'), struct.unpack_from('<2Q', raw, offset+i*16))))
                      for i, name in enumerate(PHASES))
        if any(phases[name]['ticks'] != sum(n['phases'][name] for n in nodes) for name in PHASES):
            raise ValueError('Global/node phase cycle mismatch')
        weights = dict(zip(WEIGHT_FIELDS, struct.unpack_from('<5Q', raw, offset+96)))
        if (weights['weight_pack_bytes'], weights['weight_pack_values'], weights['prepared_layers'],
                weights['runtime_weight_conversions'], phases['weight_prepare']['calls']) != (5033984, 2391664, 94, 0, 0):
            raise ValueError('Prepared weight invariant failed')
        report.update(phases=phases, weights=weights)
    total = sum(node['total'] for node in nodes)
    if (total != report['node_cycles'] or total > report['graph_cycles'] or
            report['graph_cycles']+report['input_check_cycles'] > report['model_cycles'] or
            report['model_cycles']+report['input_crc_cycles']+report['output_cycles'] > report['service_cycles']):
        raise ValueError('Inconsistent board cycle accounting')
    report.update(nodes_detail=nodes, crc=checksum, sha256=hashlib.sha256(raw).hexdigest(),
                  representative=not bool(report['options'] & 14),
                  graph_overhead_cycles=report['graph_cycles']-total)
    return report


def validate_saved_profile(profile, frame):
    raw = struct.pack('<16Q', *(profile[k] for k in FIELDS))
    for node in profile['nodes_detail']:
        raw += struct.pack('<3Q', *(node[k] for k in ('cache', 'compute', 'check')))
        if profile['version'] == 2:
            raw += struct.pack('<6Q', *(node['phases'][k] for k in PHASES))
    if profile['version'] == 2:
        raw += b''.join(struct.pack('<2Q', profile['phases'][k]['ticks'], profile['phases'][k]['calls'])
                        for k in PHASES)
        raw += struct.pack('<5Q', *(profile['weights'][k] for k in WEIGHT_FIELDS))
    raw += struct.pack('<Q', profile['crc'])
    if hashlib.sha256(raw).hexdigest() != profile['sha256']:
        raise ValueError('Saved board timing SHA256 mismatch')
    return decode_profile(raw, frame, profile['session'])
