"""V29 with and without the orientation switch EF_K31T21 (V29's slot is symmetric, so the predictions must agree up
to float32 rounding): compares data/pred/base_0.npy and data/pred/base_t21_0.npy."""
import numpy as np
a = np.load('./data/pred/base_0.npy').astype(float); b = np.load('./data/pred/base_t21_0.npy').astype(float)
print('max |diff| %.2e, relative to max |prediction| %.1e' % (np.abs(a - b).max(), np.abs(a - b).max() / np.abs(a).max()))
