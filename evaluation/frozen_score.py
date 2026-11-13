"""Unmodified trajectory scoring function. See README for explicit support inputs."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def score(z, t, c, T):
    ts = c['ts']
    assert np.isfinite(z).all() and z.shape[1] == 8 and np.all(np.diff(t) > 0)
    idx = np.clip(np.searchsorted(t, ts), 1, len(t) - 1)
    valid = (ts >= t[0]) & (ts <= t[-1]) & (t[idx] - t[idx - 1] <= 0.25)
    if not valid.all():
        return dict(complete=False, covered=int(valid.sum()), required=len(ts), first_missing_camera_frame=int(c['ids'][np.flatnonzero(~valid)[0]]))
    pos = np.column_stack([np.interp(ts, t, z[:, k]) for k in [1, 2, 3]])
    pc, gc = (pos.mean(0), c['gpos'].mean(0))
    u, s, vt = np.linalg.svd((pos - pc).T @ (c['gpos'] - gc))
    rot = vt.T @ np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)]) @ u.T
    err = (pos - pc) @ rot.T + gc - c['gpos']
    ate = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
    result = dict(complete=True, ATE_m=ate, metric=ate)
    if len(c['segments']):
        bodyR = Slerp(t, Rotation.from_quat(z[:, 4:8]))(ts).as_matrix()
        camR = bodyR @ T[:3, :3]
        camP = pos + np.einsum('nij,j->ni', bodyR, T[:3, 3])
        te = np.linalg.solve(camR[c['si']], (camP[c['sj']] - camP[c['si']])[..., None])[..., 0]
        relative = np.linalg.solve(camR[c['si']], camR[c['sj']])
        error = np.linalg.solve(relative, (c['tg'] - te)[..., None])[..., 0]
        metric = float(np.mean(np.linalg.norm(error, axis=1) / c['segments'][:, 2] * 100))
        result.update(metric=metric, t_rel_percent=metric, segments=len(c['segments']))
    return result
