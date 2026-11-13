"""Native ORB-SLAM3 stereo-inertial replay with reversible module-removal switches.

No GT input. The frozen model code is not edited: every arm only removes existing
runtime environment switches or flips the pre-built ablation switches present in
the child image. Output directories are exclusive per declared task.
"""
import argparse, json, re, subprocess, os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--data', type=Path, required=True, help='Sequence mav0 directory')
p.add_argument('--dataset', choices=['euroc', 'kitti'], required=True)
p.add_argument('--sequence', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--arm', choices=['supported_history'], required=True)
p.add_argument('--socket-name', required=True)
p.add_argument('--geometry', type=int, choices=[0, 1], default=1)
p.add_argument('--history', type=int, choices=[0, 1, 2], default=2)
p.add_argument('--binary', choices=['stereo_inertial', 'mono_inertial', 'stereo'], default='stereo_inertial')
p.add_argument('--remove', default='', help='comma-separated exact KEY=VALUE tokens to delete')
p.add_argument('--extra', default='', help='comma-separated KEY=VALUE tokens to append')
p.add_argument('--raw-stereo-log', default='')
a = p.parse_args()
R = a.root.resolve(); out = a.output.resolve(); bundle = Path(__file__).resolve().parent
meta = json.loads((bundle / 'calibration_manifest.json').read_text())
identity = json.loads((bundle / 'built_image.json').read_text())
image = identity['image']
actual = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], text=True).strip()
assert actual == identity['image_id'], (actual, identity['image_id'])
key = a.dataset + '/' + a.sequence
assert key in meta['calibrations']
assert not out.exists(), 'Output already exists; refusing overwrite'
raw = a.data.resolve()
assert all((raw / s / 'data.csv').is_file() for s in ['cam0', 'cam1', 'imu0'])
out.mkdir(parents=True)
shift = 0 if a.dataset == 'euroc' else 1000000000000
left = {l.split(',')[0] for l in (raw / 'cam0/data.csv').read_text().splitlines() if l and not l.startswith('#')}
right = {l.split(',')[0] for l in (raw / 'cam1/data.csv').read_text().splitlines() if l and not l.startswith('#')}
paired = left & right
times = []
for sensor in ['cam0', 'cam1', 'imu0']:
    dst = out / 'sensors/mav0' / sensor
    dst.mkdir(parents=True)
    lines = []
    for l in (raw / sensor / 'data.csv').read_text().splitlines():
        if not l or l.startswith('#'):
            lines.append(l); continue
        first, rest = l.split(',', 1)
        if sensor.startswith('cam') and first not in paired:
            continue
        t = int(first) + shift
        lines.append(str(t) + ',' + rest)
        if sensor.startswith('cam'):
            (dst / 'data').mkdir(exist_ok=True)
            (dst / 'data' / f'{t}.png').symlink_to(raw / sensor / 'data' / rest.strip())
            if sensor == 'cam0':
                times.append(str(t))
    (dst / 'data.csv').write_text('\n'.join(lines) + '\n')
(out / 'timestamps.txt').write_text('\n'.join(times) + '\n')
cfg = (bundle / 'author_EuRoC.yaml').read_text()
c = meta['calibrations'][key]


def scalar(k, v):
    global cfg
    cfg, n = re.subn(r'^' + re.escape(k) + r':[^\n]*', k + ': ' + str(v), cfg, flags=re.M)
    assert n == 1, k


if a.dataset == 'kitti':
    for i, cam in enumerate(c['cameras'], 1):
        for k, v in zip(['fx', 'fy', 'cx', 'cy'], cam['intrinsics']):
            scalar('Camera' + str(i) + '.' + k, v)
        for k, v in zip(['k1', 'k2', 'p1', 'p2'], cam['distortion_coefficients']):
            scalar('Camera' + str(i) + '.' + k, v)
    for k, v in [('Camera.width', c['cameras'][0]['resolution'][0]), ('Camera.height', c['cameras'][0]['resolution'][1]), ('Camera.fps', 10), ('IMU.Frequency', 100.)]:
        scalar(k, v)
    for k, mat in [('IMU.T_b_c1', c['T_body_camera0']), ('Stereo.T_c1_c2', c['T_camera0_camera1'])]:
        pattern = r'^' + re.escape(k) + r': !!opencv-matrix\s+rows: 4\s+cols: 4\s+dt: f\s+data: \[[^\]]+\]'
        value = k + ': !!opencv-matrix\n  rows: 4\n  cols: 4\n  dt: f\n  data: [' + ', '.join(format(v, '.17g') for v in mat) + ']'
        cfg, n = re.subn(pattern, lambda _: value, cfg, flags=re.M)
        assert n == 1, k
cfg += '\nloopClosing: 0\n'
(out / 'configuration.yaml').write_text(cfg)

exe = {'stereo_inertial': '/opt/ORB_SLAM3/Examples/Stereo-Inertial/stereo_inertial_euroc',
       'mono_inertial': '/opt/ORB_SLAM3/Examples/Monocular-Inertial/mono_inertial_euroc',
       'stereo': '/opt/ORB_SLAM3/Examples/Stereo/stereo_euroc'}[a.binary]
cmd = ['docker', 'run', '--init', '--rm', '--network', 'none',
       '-v', str(R) + ':' + str(R) + ':ro', '-v', str(raw) + ':' + str(raw) + ':ro', '-v', str(out) + ':/output:rw',
       '-w', '/output', image, exe,
       '/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt', '/output/configuration.yaml',
       '/output/sensors', '/output/timestamps.txt']
extra = ['--cpus=' + os.environ.get('VIO_NATIVE_CPUS', '6'), '--memory=24g', '-v', '/tmp/vio_source_map_bridge:/bridge',
         '-e', 'VIO_RECTIFIED_STEREO_SOCKET=/bridge/' + a.socket_name,
         '-e', 'VIO_MAP_CANDIDATE_LOG=/output/map_candidates.csv',
         '-e', 'VIO_MAP_SOURCE_ACTIVATION=/output/source_activation.csv',
         '-e', 'VIO_SOURCE_CLOCK_SHIFT=' + ('1000' if a.dataset == 'kitti' else '0')]
extra += ['-e', 'VIO_A24_LOG=/output/a24_state_flags.csv']
extra += ['-e', 'VIO_A32_LOG=/output/a32_support.csv']
extra += ['-e', 'VIO_ABLATION_GEOMETRY=' + str(a.geometry), '-e', 'VIO_ABLATION_HISTORY=' + str(a.history),
          '-e', 'VIO_COMPONENT_ABLATION_LOG=/output/components.csv']
extra += ['-e', 'VIO_A61_LOG=/output/a61.csv', '-v', str(out.parent / 'learned_frontend/a57_grid') + ':/grid:ro']
extra += ['-e', 'VIO_A59_LOG=/output/a59.csv', '-e', 'VIO_A53_LOG=/output/a53.csv']
extra += ['-e', 'VIO_A32_COMPLETE_SUPPORT=1']
extra += ['-e', 'VIO_SOURCE_HISTORY_LOG=/output/source_history.csv']
extra += ['-e', 'VIO_TIME_NORMALIZED_INIT=1', '-e', 'VIO_INITIALIZATION_CADENCE_LOG=/output/initialization_cadence.csv']
extra += ['-e', 'VIO_MAP_CANDIDATES=1', '-e', 'VIO_A1_FLOW_LOG=/output/a1_source_flow.csv',
          '-e', 'VIO_A1_BIRTH_LOG=/output/a1_birth', '-e', 'VIO_A1_CURRENT_LOG=/output/a1_current_measurements.csv']
extra += ['-e', 'VIO_A3_OWNER_LOG=/output/a3_seed_ownership.csv', '-e', 'VIO_A3_HISTORY_LOG=/output/a3_history.csv',
          '-e', 'VIO_A3_QUALITY_LOG=/output/a3_quality.csv']
extra += ['-e', 'VIO_A4_STATE_LOG=/output/a4_state.csv', '-e', 'VIO_A4_ENTRY_LOG=/output/a4_entry.csv']
extra += ['-e', 'VIO_A4_CROSS_KF=1']
extra += ['-e', 'VIO_A5_RECORD_LOG=/output/a5_records.csv', '-e', 'VIO_A5_SNAPSHOT_LOG=/output/a5_snapshots.csv',
          '-e', 'VIO_A5_TRANSFER_LOG=/output/a5_transfer.csv', '-e', 'VIO_A5_MAP_LOG=/output/a5_map.csv',
          '-e', 'VIO_A5_RECEIPT_LOG=/output/a5_receipt.csv']
extra += ['-e', 'VIO_A6_EVENT_LOG=/output/a6_events.csv']
extra += ['-e', 'VIO_RESEARCH_SEED=' + os.environ['VIO_RESEARCH_SEED'],
          '-e', 'VIO_A6_KF_LOG=/output/a6_keyframes.csv', '-e', 'VIO_A6_TRACK_LOG=/output/a6_tracking_events.csv',
          '-e', 'VIO_A7_INIT_LOG=/output/a7_initialization.csv', '-e', 'VIO_A7_PUBLISH_LOG=/output/a7_publication.csv',
          '-e', 'VIO_A7_REFERENCE_LOG=/output/a7_reference.csv']
extra += ['-e', 'VIO_FRAME_COMMIT_BARRIER=1', '-e', 'VIO_FRAME_BARRIER_LOG=/output/frame_commit_barrier.csv']
extra += ['-e', 'VIO_A12_BRANCH_LOG=/output/a12_branches.csv']
extra += ['-e', 'VIO_A14_PROPOSE=1', '-e', 'VIO_A14_ASSOCIATION_LOG=/output/a14_associations.csv']
extra += ['-e', 'VIO_A15_BUNDLE_LOG=/output/a15_bundles.csv']
extra += ['-e', 'VIO_A16_PROPOSE=1', '-e', 'VIO_A16_LOG=/output/a16_route.csv', '-e', 'VIO_A16_SAMPLE_LOG=/output/a16_samples.csv']
extra += ['-e', 'VIO_A16_ACTIVE=1']
extra += ['-e', 'VIO_A16_HISTORY=1']
extra += ['-e', 'VIO_A17_IDENTITY_LOG=/output/a17_identity.csv']
extra += ['-e', 'VIO_SHUTDOWN_LOG=/output/shutdown_boundary.csv']
extra += ['-e', 'VIO_A17_ANCHOR=1']
extra += ['-e', 'VIO_A18_EARLY=1', '-e', 'VIO_A22_LOG=/output/a22_history.csv']
extra += ['-e', 'VIO_A22_ACTIVE=1']

remove = {x.strip() for x in a.remove.split(',') if x.strip()}
if remove:
    filtered = []
    i = 0
    while i < len(extra):
        if extra[i] == '-e' and i + 1 < len(extra) and extra[i + 1] in remove:
            i += 2
            continue
        filtered.append(extra[i])
        i += 1
    extra = filtered
    for token in remove:
        assert token not in extra, token
for token in [x.strip() for x in a.extra.split(',') if x.strip()]:
    key_, value = token.split('=', 1)
    assert key_.startswith('VIO_')
    extra += ['-e', token]
if a.raw_stereo_log:
    extra += ['-e', 'VIO_ABLATION_RAW_STEREO_LOG=' + a.raw_stereo_log]
cmd[cmd.index(image):cmd.index(image)] = extra
(out / 'execution_manifest.json').write_text(json.dumps(dict(
    command=cmd, dataset=a.dataset, sequence=a.sequence,
    original_left=len(left), original_right=len(right), physical_pairs=len(paired),
    clock_shift_ns=shift, GT_input=False, image_id=actual, binary=a.binary,
    geometry=a.geometry, history=a.history, removed=sorted(remove),
    added=[x for x in [t.strip() for t in a.extra.split(',') if t.strip()]],
    scope='Full stereo+IMU; native final no-loop export; module switches only, no code edit'), indent=2))
with (out / 'native.log').open('w') as f:
    run = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=21600)
(out / 'execution.json').write_text(json.dumps(dict(returncode=run.returncode, trajectory_exists=(out / 'CameraTrajectory.txt').exists(), accepted=False), indent=2))
raise SystemExit(run.returncode)
