#!/bin/sh
track_pillars medium_disp.csv medium_points.csv "0 1"
analyze_mechanics medium_disp.csv "0 1 2 3 4 5 6 7 8 9 10"
visualize_mechanics medium_disp.csv "1 2 3 4 5 6 7 8 9"
