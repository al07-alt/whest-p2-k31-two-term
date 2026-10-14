"""Summary numbers of the two-term law over networks 0-22 (from law23.py output data/law23_0_22.npz) and of the
fixed-coefficient out-of-sample R2 (law_oos.py output data/law_oos.npz)."""
import numpy as np
z = np.load('./data/law23_0_22.npz'); nets = range(23)
c = np.array([z[f'c_{i}'] for i in nets])            # (net, layer 1..14, [b1, b2, R2rad, R2two, b1fix, brad])
u = np.array([z[f'u_{i}'] for i in nets])            # (net, |alpha| bin) radial row scale / global, l = 14
print('b2, layers 4-14, all 23 nets: min %.2f max %.2f' % (c[:, 3:, 1].min(), c[:, 3:, 1].max()))
print('median b2 over nets per layer 1..14:', np.round(np.median(c[:, :, 1], 0), 2))
print('median b1 over nets per layer 1..14:', np.round(np.median(c[:, :, 0], 0), 4))
print('l=14 R2 radial %.2f-%.2f, two-term %.2f-%.2f' % (c[:, -1, 2].min(), c[:, -1, 2].max(), c[:, -1, 3].min(), c[:, -1, 3].max()))
for k, lab in enumerate(['|a|<0.75', '0.75-1.5', '1.5-2.5', '2.5-4', '>4']):
    print('l=14 radial row scale / global, %-8s median %.2f range %.2f-%.2f' % (lab, np.median(u[:, k]), u[:, k].min(), u[:, k].max()))
o = np.load('./data/law_oos.npz'); r2 = o['r2']        # (net 3..22, layer, [fixed-joint, fixed-refit, LS-two, LS-rad])
gap = r2[:, :, 2] - r2[:, :, 0]
print('fixed-coefficient (joint table, b2=1) R2 at l=14: %.2f-%.2f (median %.2f); per-net LS two-term median %.2f' % (
    r2[:, -1, 0].min(), r2[:, -1, 0].max(), np.median(r2[:, -1, 0]), np.median(r2[:, -1, 2])))
print('gap LS-two minus fixed, l=14: %.3f-%.3f (median %.3f); all layers: %.3f-%.3f (median %.3f)' % (
    gap[:, -1].min(), gap[:, -1].max(), np.median(gap[:, -1]), gap.min(), gap.max(), np.median(gap)))
