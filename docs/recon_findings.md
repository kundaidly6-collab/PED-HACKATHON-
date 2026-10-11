# Reconnaissance findings — real component geometry

Source: the first real part image supplied (galvanised steel mounting plate, studio
photo, white backdrop). Analysed with `scripts/recon_image.py`.

**Status: the image file did not reach the sandbox**, so the numbers below were
confirmed on a structurally faithful stand-in (`/tmp/recon_test/standin.png`: same
backdrop, plate, hole layout, three bore appearances and one weld ring) to validate the
tool. Re-run on the real file to replace them:

```bash
python scripts/recon_image.py path/to/real_part.png --out runs/recon/part.png
```

## What the component is

A square galvanised mounting plate with a large recessed (pressed) panel, formed ribs
inside the recess, and mounting holes around the perimeter flange. One small circular
weld/stud feature sits inside the recess near its lower edge.

## Geometry (stand-in measurements)

| property | value | consequence |
|---|---|---|
| frame | 1560×945 px | typical enthusiast/phone capture; expect the real set to vary |
| component | ~960×840 px, 62%×89% of frame | the part dominates the frame — trim it, do not detect on clutter |
| perimeter holes | 11, **~48 px diameter** (3.1% of frame width) | large, easy targets |
| weld feature | ~53 px outer, ~26 px core | similar scale to the holes; harder to separate by size |
| at imgsz 640 | a 48 px hole → **~20 px** | comfortably above the stride-32 floor; **640 is defensible**, no tiling needed |
| at imgsz 1280 | same hole → ~39 px | only worth it if Spot turns out to be the ~10 px case |

`imgsz=640` is the right starting point. Escalate to 1024/1280 only if the real Spot
instances are much smaller than the ~50 px stand-in feature.

## The finding that matters most: Hole has at least three appearances

Detected bores break down as:

| appearance | count | what it is |
|---|---|---|
| bright / through | 3 | backdrop visible through the bore (top-left, top-right, bottom-right) |
| dark bore (shadowed) | 5 | unlit bore, near-black |
| coloured bore (substrate visible) | 3 | warm/copper tone showing material behind the plate |
| mid-tone bore | 2 | intermediate / shadowed |

These are the **same class**, rendered as near-white, near-black, and saturated warm
tones. Two consequences:

1. **Label coverage.** If the annotator boxes only the obvious dark bores, the model
   learns "dark circle" and misses every bright and coloured one. All three appearances
   must be present in the training set with correct labels.
2. **Augmentation limit.** Aggressive HSV/colour augmentation is actively dangerous here:
   a hue shift can move a "bright through-hole" toward "shadowed bore", teaching the
   model that the appearance is noise and the class is unstable. Keep `hsv_s`/`hsv_v`
   moderate (the current defaults are 0.5/0.4 — consider 0.3/0.3) and prefer geometric
   augmentation (`scale`, `translate`, mild `mosaic`), which is genuinely useful for a
   fixed camera.

## Annotation ambiguity to settle with a human

The interior weld ring reads as a **round** feature with a **bright rim and dark core** —
the same topology as a bore. Position is the only clean discriminator available:

- **perimeter band** (flange) → mounting holes → likely `Hole`
- **panel interior** → this weld/stud → likely `Spot`

`recon_image.py` reports the zone and refuses to assign a class. **Confirm with the
organisers which class the interior circular feature belongs to before labelling**, since
every image in the set will share this layout and getting it backwards would invert the
class mapping across the entire dataset — the single most expensive mistake available here.

## Counts to expect per image

If the real set shares this design: **~11 Hole** instances (perimeter) and **1–4 Spot**
instances (interior), i.e. a strongly Hole-tilted class distribution. Note this is the
*opposite* imbalance from the synthetic demo, where Spot outnumbered Hole. Expect the
minority class (Spot) to gate mAP@45 and plan annotation effort accordingly.
