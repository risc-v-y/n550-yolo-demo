"""Decode CPU-cycle counters published with PCIe DONE; Python 3.8, stdlib only."""
import hashlib
import struct
import zlib

PROFILE_MAGIC = 0x3152465036325955
PROFILE_BYTES = 8152
FIELDS = ('magic version bytes frame nodes completed options state model_cycles '
          'graph_cycles input_check_cycles service_cycles input_crc_cycles output_cycles '
          'node_cycles session').split()


def decode_profile(raw, frame, session):
    if len(raw) != PROFILE_BYTES:
        raise ValueError('Board timing length mismatch')
    checksum = struct.unpack_from('<Q', raw, len(raw)-8)[0]
    if checksum != zlib.crc32(raw[:-8]):
        raise ValueError('Board timing CRC mismatch')
    report = dict(zip(FIELDS, struct.unpack_from('<16Q', raw)))
    if (report['magic'], report['version'], report['bytes']) != (PROFILE_MAGIC, 1, PROFILE_BYTES):
        raise ValueError('Unsupported board timing layout')
    if (report['frame'], report['session']) != (frame, session):
        raise ValueError('Board timing frame/session mismatch')
    if (report['nodes'], report['completed'], report['state']) != (334, 334, 1):
        raise ValueError('Incomplete board timing')
    if report['options'] & ~14:
        raise ValueError('Unknown board timing options')
    nodes = []
    for i in range(334):
        cache, compute, check = struct.unpack_from('<3Q', raw, 128+i*24)
        nodes.append(dict(node=i, cache=cache, compute=compute, check=check,
                          total=cache+compute+check))
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
    raw += b''.join(struct.pack('<3Q', *(node[k] for k in ('cache', 'compute', 'check')))
                    for node in profile['nodes_detail'])
    raw += struct.pack('<Q', profile['crc'])
    if hashlib.sha256(raw).hexdigest() != profile['sha256']:
        raise ValueError('Saved board timing SHA256 mismatch')
    return decode_profile(raw, frame, profile['session'])
