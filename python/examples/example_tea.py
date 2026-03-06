"""
example_tea.py — End-to-end demo of the Python TEA class.

Demonstrates: create, read, append, discontinuity handling, refresh.
Mirrors the MATLAB example_brew_and_drink.m.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tea import TEA


def main():
    # 1. Create a TEA file with regular continuous data
    print("=" * 60)
    print("1. Create TEA file")
    SR = 1000
    t = np.arange(5000) / SR
    samples = np.random.randn(5000, 3)

    tea = TEA('demo.mat', SR, True, t_units='s')
    tea.write(t, samples)
    print(tea.info())

    # 2. Read by time range
    print("\n2. Read by time range")
    Data, t_out, _ = tea.read(channels=[1, 3], t_range=(1.0, 2.5))
    print(f"Read {Data.shape[0]} samples, channels [1,3], t=[{t_out[0]:.3f}, {t_out[-1]:.3f}]")

    # 3. Read by sample range
    print("\n3. Read by sample range")
    Data2, t_out2, _ = tea.read(s_range=(99, 499))
    print(f"Read samples 99-499: {Data2.shape[0]} samples")

    # 4. Append more time samples
    print("\n4. Append time samples")
    t2 = t[-1] + np.arange(1, 5001) / SR
    s2 = np.random.randn(5000, 3)
    tea.write(t2, s2)
    print(f"After append: N={tea.N}, C={tea.C}")

    # 5. Append channels
    print("\n5. Append channels")
    tea.write_channels(np.random.randn(10000, 2), ch_map=[4, 5])
    print(f"After channel append: N={tea.N}, C={tea.C}, ch_map={tea.ch_map}")

    # 6. Discontinuous data
    print("\n6. Discontinuous data")
    t_disc = np.concatenate([np.arange(1000) / SR, 3.0 + np.arange(1000) / SR])
    s_disc = np.random.randn(2000, 1)
    tea_disc = TEA('demo_disc.mat', SR, True, t_units='s')
    tea_disc.write(t_disc, s_disc)

    _, _, disc_info = tea_disc.read()
    print(f"Discontinuous: {disc_info['is_discontinuous']}, gaps: {disc_info['disc'].shape[0]}")

    # 7. Irregular data
    print("\n7. Irregular data")
    t_irr = np.sort(np.random.rand(200) * 100)
    tea_irr = TEA('demo_irreg.mat', None, False)
    tea_irr.write(t_irr, np.random.randn(200, 2))
    print(f"Irregular: N={tea_irr.N}")

    # 8. Cleanup
    print("\n8. Cleanup")
    for fname in ['demo.mat', 'demo_disc.mat', 'demo_irreg.mat']:
        if os.path.exists(fname):
            os.remove(fname)
    print("Done!")


if __name__ == '__main__':
    main()
