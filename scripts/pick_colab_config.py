#!/usr/bin/env python3
"""Pick a Colab config from the attached GPU."""

from ntv3_crop.hardware import recommend_config

if __name__ == "__main__":
    print(recommend_config())
