#!/bin/sh
track_pillars small_disp.csv small_points.csv "0 1"
analyze_mechanics small_disp.csv "0 1 2 3 4 5 6 7 8 9 10"
visualize_mechanics small_disp.csv "1 2 3 4 5 6 7 8 9"
