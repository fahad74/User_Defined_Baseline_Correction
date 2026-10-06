# Baseline Correction Tool (FTIR-ready)

A simple desktop GUI (Python + tkinter + matplotlib) for loading XY spectral
data, reversing the X axis (for FTIR spectra shown 4000 → 400 cm⁻¹),
picking a baseline interactively with your mouse, editing baseline points,
subtracting the baseline, and exporting the result.

## 1. Install

Requires Python 3.8+. This uses `tkinter`, which ships with most standard
Python installs (on some Linux distros you may need to install it
separately, e.g. `sudo apt install python3-tk`).

```bash
pip install -r requirements.txt
```

## 2. Run

```bash
python ftir_baseline_app.py
```

A sample dataset, `sample_ftir_data.csv`, is included so you can try it
out immediately (File → Browse Data).

## 3. How to use

1. **Browse Data** — load a two-column XY file (`.csv`, `.txt`/`.dat`, or
   `.xlsx`). The app auto-detects delimiters and skips a header row if
   present.
2. **Reverse X Axis (FTIR)** — check this box to display wavenumber from
   high to low (4000 → 400 cm⁻¹), as is conventional for FTIR spectra.
   This only affects the display/export order, not the underlying values.
3. **Target # baseline points** — set how many points you plan to pick.
4. **Click on Plot to Add Points** — enable this, then click directly on
   the spectrum at each point you want to anchor the baseline to. Picking
   automatically turns off once you hit your target count (you can
   re-enable it any time to add more).
   - *Note:* while the matplotlib toolbar's pan/zoom tool is active,
     clicks are used for panning/zooming instead of adding points — turn
     those off (click the tool button again) before picking baseline
     points.
5. **Edit points** — double-click a row in the "Baseline Points" table to
   change its X or Y value. Select a row and click "Remove Selected
   Point" to delete it. Use the "Add point manually" box to type in exact
   coordinates.
6. **Subtract Baseline** — once you have at least 2 points, click this to
   linearly interpolate a baseline through your points across the full
   data range and subtract it. The corrected curve is drawn in green.
7. **Export Data** — save the results as `.csv`, `.txt` (tab-separated),
   or `.xlsx`. The exported file includes the raw X/Y, the fitted
   baseline, and the corrected Y (when computed).

## Notes

- Baseline points don't need to be at existing data samples — click
  anywhere on/near the curve; the baseline is a fresh interpolation.
- Points outside the picked baseline range use the nearest edge baseline
  value (flat extrapolation), so make sure to place points near both ends
  of your spectrum for realistic results.
- All computations are the underlying X values (never reversed) — the
  "Reverse X Axis" option only affects how things are displayed and how
  rows are ordered on export.
