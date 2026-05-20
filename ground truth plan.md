# Ground-truth acquisition plan

This is the critical path. Every code module can be written in a day; yield
labels for Zambian maize can take weeks to obtain and define what model is
even possible. Read this before writing `dataset.py` or finalising the GEE
export seasons — it changes both.

---

## Headline finding (act on this first)

The most recent **Rural Agricultural Livelihoods Survey (RALS)** wave is
**2019**, which records the **2018/19** agricultural season. The Sentinel-2
seasons currently in `config.py` are 2022/23, 2023/24, 2024/25. **They do not
overlap.** A supervised yield model needs labels and imagery from the *same*
season.

Two consequences:

1. **The GEE export must add the 2018/19 season.** Sentinel-2 has reliable
   coverage from 2017 onward, so 2018/19 is feasible. RALS 2015 (2014/15
   season) is *not* usable — Sentinel-2 was not operational then.
2. The 2022–2025 exports are still useful — for the operational/inference
   demo and for CFS district-level weak labels — but they are **not** the
   supervised training set.

This is why ground truth comes before more code: it rewrites the data
contract.

---

## The two hard constraints

### Temporal
RALS 2019 → pair with Sentinel-2 **2018/19** composites. Add `2018_19`
(window ~2018-10-15 to 2019-06-15) to `SEASONS` in `config.py` and to
`gee/fg_export.py`, then export the three sub-AOIs for it. This is the
training backbone season.

### Spatial
RALS gives a **household GPS point and a plot area in hectares** — not a
field boundary polygon. Public microdata often offsets the point to the
enumeration-area (EA) level for privacy. You therefore cannot extract a
precise field patch.

Modelling implication: train at the spatial unit the labels actually
support. Aggregate Sentinel-2 within a buffer (e.g. EA polygon, or a radius
sized to typical plot area) around the household/EA point and predict
yield for that unit. The thesis must state plainly that this is
**cluster/EA-level maize yield estimation**, not field-polygon-level. This
changes `dataset.py`: the unit of analysis is a buffered point, not a tile.

---

## Source ladder

Ordered by label quality and realism for a thesis timeframe.

### Tier 1 — RALS 2019 (the supervised backbone)
Producer: IAPRI, with Zambia Statistics Agency and MoA. Section 2 (Farm Land
and Use) gives plot area; production data gives output → derive maize yield
(kg/ha) per plot. Panel survey, nationally representative of small/medium
farms (<20 ha).

Access: microdata is cataloged on the IHSN Survey Catalog
(`catalog.ihsn.org`, Reference ID series `ZMB_*_RALS_*`) and the World Bank
Microdata Library. Both require a free account and a data-request /
statement-of-use form. RALS 2019 specifically may require a direct request to
IAPRI — verify the current process; as a Lusaka-based researcher you likely
have a faster route through IAPRI directly than the catalog queue.

Expected friction: account + request approval can take days to a few weeks.
**Submit the request today.** This is the long pole.

### Tier 2 — Crop Forecast Survey (sanity + weak labels)
Producer: MoA / Zambia Statistics Agency, annual. District-level maize area
and production for recent years, including 2022–2025. Use as: (a) a sanity
bound on Tier 1 derived yields, (b) optional district-level weak supervision
for the 2022–2025 exports you already have. Coarse — not a substitute for
RALS.

### Tier 3 — Primary collection (gold validation, small n)
A small, high-quality validation set you control. Scope it to thesis-feasible:
30–50 plots across Chilanga, Kafue, Chongwe for the most recent completed
season, each with a GPS point, plot area, and farmer-reported (or measured)
harvest weight. Route: agronomy/extension contacts in the three districts,
or a camera-club / cooperative partner. This cannot be done retroactively, so
if you want it for 2025/26, the protocol has to go out before the next
harvest. Treat as a stretch goal — RALS 2019 is the backbone.

### Tier 4 — Malawi LSMS-ISA (transfer / augmentation fallback)
Zambia is not an LSMS-ISA country; Malawi is, and it's georeferenced,
multi-wave, and the same rain-fed maize system next door. If the RALS 2019
sample falling inside Lusaka Province is too thin for training, use Malawi
LSMS-ISA for pretraining or augmentation, then fine-tune on the Zambian
sample. Instant download from the World Bank Microdata Library — no approval
delay. Good insurance.

### Sanity only — FAOSTAT
National/provincial maize production and area. Use only to check that derived
yields are in a plausible range (Zambia smallholder maize is broadly ~1.5–3.0
t/ha rain-fed; treat as a sniff test, not a label).

---

## Concrete actions

| # | Action | Where | Start | Latency |
|---|---|---|---|---|
| 1 | Request RALS 2019 microdata | IAPRI direct + IHSN/World Bank catalog | **today** | days–weeks |
| 2 | Download CFS district maize series 2018–2025 | ZamStats / MoA publications | this week | hours |
| 3 | Add `2018_19` season to `config.py` + `fg_export.py`, export 3 sub-AOIs | local | after #1 sent | ~3 h compute |
| 4 | Download Malawi LSMS-ISA (fallback) | World Bank Microdata Library | this week | instant |
| 5 | Decide go/no-go on primary collection; if go, draft a one-page field protocol | — | this week | — |
| 6 | FAOSTAT Zambia maize pull for range-check | fao.org/faostat | this week | minutes |

Actions 1 and 3 are the blockers. 2, 4, 6 are quick and de-risk the project.

---

## Pipeline changes this forces

- `config.py`: add `"2018_19"` to `SEASONS`; document that 2018/19 is the
  supervised season and 2022–2025 are inference/weak-label.
- `gee/fg_export.py`: add the `2018_19` date window; re-export sub-AOIs.
- `src/farmers_guide/data/ground_truth.py`: parse RALS plot area + production
  → `yield_kg_ha`, keyed by EA/household ID with its GPS point.
- `src/farmers_guide/data/dataset.py`: unit of analysis is a **buffered
  point**, not a tile. Aggregate the 14-band Sentinel-2 stack within the
  buffer; the satellite branch consumes the temporal sequence of those
  aggregates (or a fixed patch centred on the point if EA offset is small
  enough — decide once the RALS geovariables are in hand).

---

## Minimum viable label set + fallback ladder

A defensible thesis needs, in priority order:

1. **Best case:** RALS 2019 plot-level maize yields for Lusaka Province
   (+ Chongwe/Kafue/Chilanga) joined to Sentinel-2 2018/19. n in the low
   hundreds is workable for cluster-level modelling.
2. **If the Lusaka-Province RALS sample is thin:** add Malawi LSMS-ISA for
   pretraining, fine-tune on the Zambian sample.
3. **If RALS access stalls past the thesis runway:** fall back to CFS
   district-level weak supervision on the 2022–2025 exports + the Tier-3
   primary set as the only field-resolution validation, and reframe the
   contribution as a district-scale forecasting method validated on a small
   gold set. Weaker, but still a complete thesis.

State whichever path you land on explicitly in the methodology — examiners
respect an honest data-limitation section far more than an overclaimed one.

---

## Today

Submit the RALS 2019 request (action 1) and pull CFS + FAOSTAT (actions 2,
6) — these need no code and unblock everything. In parallel, Claude Code
builds `hdf5_builder.py`, which is season-agnostic and not blocked by any of
this.
