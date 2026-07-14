# Changelog

## [1.4.1](https://github.com/rocknroll17/QR-Bloom/compare/v1.4.0...v1.4.1) (2026-07-14)


### Bug Fixes

* lowercase Docker image name in release workflow ([#77](https://github.com/rocknroll17/QR-Bloom/issues/77)) ([7fd230e](https://github.com/rocknroll17/QR-Bloom/commit/7fd230e3c6a89d14dbe0577849cb2582a21a1a23))

## [1.4.0](https://github.com/rocknroll17/QR-Bloom/compare/v1.3.1...v1.4.0) (2026-07-14)


### Features

* **pages:** grow-in animation — voxels rise from the trunk (BFS) on generate ([#68](https://github.com/rocknroll17/QR-Bloom/issues/68)) ([0d09643](https://github.com/rocknroll17/QR-Bloom/commit/0d09643d375337432f9a96780bafbf4d2fe66645))


### Bug Fixes

* **diffusion:** make occupancy pos_weight a per-voxel map instead of per-sample scalar ([#70](https://github.com/rocknroll17/QR-Bloom/issues/70)) ([1b7d567](https://github.com/rocknroll17/QR-Bloom/commit/1b7d567b3701fb283a5d92e7cc90a052ccffb1ad))

## [1.3.1](https://github.com/rocknroll17/QR-Bloom/compare/v1.3.0...v1.3.1) (2026-06-09)


### Documentation

* attribution — permission granted (non-commercial) + vercel link + credit Enzo ([#66](https://github.com/rocknroll17/QR-Bloom/issues/66)) ([530f671](https://github.com/rocknroll17/QR-Bloom/commit/530f6710504c6596a6b09065f9dccc3aa51febe0))
* credit Grow-Voxly for tree-gen algorithm + NOTICE ([#64](https://github.com/rocknroll17/QR-Bloom/issues/64)) ([1964b82](https://github.com/rocknroll17/QR-Bloom/commit/1964b82b848f1518081a1bba03ba3f676359ee1e))

## [1.3.0](https://github.com/rocknroll17/QR-Bloom/compare/v1.2.0...v1.3.0) (2026-06-09)


### Features

* **pages:** persist weights in Cache API keyed by content hash ([#62](https://github.com/rocknroll17/QR-Bloom/issues/62)) ([2bfc013](https://github.com/rocknroll17/QR-Bloom/commit/2bfc013702c85b1114819e3b12fed9e53c5ea4c7))


### Bug Fixes

* **pages:** bypass HTTP cache on download retry (stale 403) ([#61](https://github.com/rocknroll17/QR-Bloom/issues/61)) ([7a75399](https://github.com/rocknroll17/QR-Bloom/commit/7a75399cdf11b86f5a0dd202447f634eee1294b9))
* **pages:** retry weight downloads + gate Generate on real readiness ([#59](https://github.com/rocknroll17/QR-Bloom/issues/59)) ([a5ff4fa](https://github.com/rocknroll17/QR-Bloom/commit/a5ff4fab1273d06b43f46908cd67751d789ef59a))
* replace global inference lock with per-version locks ([#63](https://github.com/rocknroll17/QR-Bloom/issues/63)) ([9b8ba76](https://github.com/rocknroll17/QR-Bloom/commit/9b8ba765e2b59219ba55182cf7adaba803efce07))

## [1.2.0](https://github.com/rocknroll17/QR-Bloom/compare/v1.1.9...v1.2.0) (2026-06-06)


### Features

* **pages:** run the real model in-browser on WebGPU (tf.js) ([#52](https://github.com/rocknroll17/QR-Bloom/issues/52)) ([647d71b](https://github.com/rocknroll17/QR-Bloom/commit/647d71b1ac3849fa70c83b7613f9b14b29815eeb))


### Bug Fixes

* **scripts:** close files explicitly in export_for_tfjs (CodeQL) ([#54](https://github.com/rocknroll17/QR-Bloom/issues/54)) ([26a4466](https://github.com/rocknroll17/QR-Bloom/commit/26a4466d7580bb90292942fc33fc60105122b83d))
* **templates:** strip stray tags at end of embed.html ([#57](https://github.com/rocknroll17/QR-Bloom/issues/57)) ([d8d47aa](https://github.com/rocknroll17/QR-Bloom/commit/d8d47aa4232a96133b7c099302fd91b087120d5d))


### Documentation

* **readme:** point Try it at the in-browser WebGPU demo ([#55](https://github.com/rocknroll17/QR-Bloom/issues/55)) ([6879559](https://github.com/rocknroll17/QR-Bloom/commit/68795597fa6e01be6d31c10683e5412c3d15a157))

## [1.1.9](https://github.com/rocknroll17/QR-Bloom/compare/v1.1.8...v1.1.9) (2026-05-28)


### Features

* **pages:** run the trained diffusion model in the browser ([#43](https://github.com/rocknroll17/QR-Bloom/issues/43)) ([0d0292e](https://github.com/rocknroll17/QR-Bloom/commit/0d0292e800ec71fb03ea7db34c382670a983073c))


### Bug Fixes

* **pages:** WASM-only inference, drop WebGPU EP ([#45](https://github.com/rocknroll17/QR-Bloom/issues/45)) ([063df78](https://github.com/rocknroll17/QR-Bloom/commit/063df786c477d1c7e4df8b3df21c5f790d407985))


### Chores

* force next release to 1.1.9 ([#49](https://github.com/rocknroll17/QR-Bloom/issues/49)) ([45487bd](https://github.com/rocknroll17/QR-Bloom/commit/45487bd872b3ad63eb925fc353cf09bbb54b54e1))

## [1.1.8](https://github.com/rocknroll17/QR-Bloom/compare/v1.1.7...v1.1.8) (2026-05-27)


### Bug Fixes

* **treegen:** narrow palm augmentation variance ([#41](https://github.com/rocknroll17/QR-Bloom/issues/41)) ([a340f05](https://github.com/rocknroll17/QR-Bloom/commit/a340f05d98c064b290eb3c7c8f41eb20787f0610))

## [1.1.7](https://github.com/rocknroll17/QR-Bloom/compare/v1.1.6...v1.1.7) (2026-05-27)


### Bug Fixes

* **treegen:** slim palm trunk + truncated-normal augmentation ([#31](https://github.com/rocknroll17/QR-Bloom/issues/31)) ([4a83bcb](https://github.com/rocknroll17/QR-Bloom/commit/4a83bcb435ee7d57d182958b33f759cd71bd9fcc))


### Bug Fixes

* **deploy:** pin to release tag instead of :latest ([#29](https://github.com/rocknroll17/QR-Bloom/issues/29)) ([6fd84ae](https://github.com/rocknroll17/QR-Bloom/commit/6fd84ae0baaee510ca444c21b064465377d31b21))
* **treegen:** slim palm trunk + truncated-normal augmentation ([#34](https://github.com/rocknroll17/QR-Bloom/issues/34)) ([25d1e0e](https://github.com/rocknroll17/QR-Bloom/commit/25d1e0e56ac510b5eb4af2ffad4679ac590707a0))

## [1.1.6](https://github.com/rocknroll17/QR-Bloom/compare/v1.1.5...v1.1.6) (2026-05-27)


### Bug Fixes

* **docker:** ship the docs/ directory in the image ([#27](https://github.com/rocknroll17/QR-Bloom/issues/27)) ([290a84d](https://github.com/rocknroll17/QR-Bloom/commit/290a84d826873c3879926ceb5541df67a6d59203))

## [1.1.5](https://github.com/rocknroll17/QR-Bloom/compare/v1.1.4...v1.1.5) (2026-05-27)


### Bug Fixes

* **deploy:** retry docker pull while release.yml is still publishing ([#26](https://github.com/rocknroll17/QR-Bloom/issues/26)) ([eee2b9e](https://github.com/rocknroll17/QR-Bloom/commit/eee2b9efe0155a2b50e6b2976cc8db42e1a548a3))
* **viewer:** restore right-click pan in tree mode ([#24](https://github.com/rocknroll17/QR-Bloom/issues/24)) ([4b44fbe](https://github.com/rocknroll17/QR-Bloom/commit/4b44fbe3b26d990f32f55d5e5501e05ec203ffa6))

## [1.1.4](https://github.com/rocknroll17/QR-Bloom/compare/v1.1.3...v1.1.4) (2026-05-27)


### Documentation

* distinguish trained-model demo from procedural preview ([#20](https://github.com/rocknroll17/QR-Bloom/issues/20)) ([43dc46e](https://github.com/rocknroll17/QR-Bloom/commit/43dc46e20ff13d48e35d915b2d06ac16bbcfcb63))

## [1.1.3](https://github.com/rocknroll17/QR-Bloom/compare/v1.1.2...v1.1.3) (2026-05-27)


### Bug Fixes

* **gallery:** close CodeQL path-injection and info-exposure alerts ([#13](https://github.com/rocknroll17/QR-Bloom/issues/13)) ([5e91990](https://github.com/rocknroll17/QR-Bloom/commit/5e919900cd8196e4dbaf6cd73f9275baf0e35a71))


### UI

* **gallery:** slim down the post-generation status line ([#14](https://github.com/rocknroll17/QR-Bloom/issues/14)) ([e04996e](https://github.com/rocknroll17/QR-Bloom/commit/e04996e57cf109f425df4b22a8a23fb58eeffd3d))
