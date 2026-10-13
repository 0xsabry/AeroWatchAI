# AeroWatch AI — Test Artifact Provenance, Licensing & Dataset Cards

This document provides complete provenance, license terms, attribution requirements, and checksums for all media artifacts in `tests/artifacts/`.

## 1. Compliance and Scope Declaration

All artifacts in this test suite represent **civilian aerial objects** and **peaceful civilian environments**:
- **Civilian Multirotor Drones** (commercial quadcopters, hexacopters, surveying UAVs)
- **Civilian Aircraft** (commercial passenger airliners, regional transport planes)
- **Birds in Flight** (avian wildlife in natural airspace)
- **Civilian Balloons** (high-altitude atmospheric and hot air balloons)
- **Natural Background Negatives** (sky, clouds, water surface, tree foliage, marine decks)

> **Civilian Scope Notice**: No weapon, targeting, missile tracking, interception logic, or military sub-type recognition is included or permitted. All aircraft detections are strictly categorized as generic `Aircraft`.

---

## 2. Dataset Sources & Licenses

| Dataset / Collection | Primary Source & Repository | License | Permitted Use |
| :--- | :--- | :--- | :--- |
| **Rigoleto Drone Detection** | Hugging Face: `Rigoleto/drone-detection` | CC-BY 4.0 | Non-commercial CV research, evaluation & benchmarking |
| **Matisdsp Drone & Bird** | Hugging Face: `matisdsp/drone-bird-detection-dataset` | CC-BY 4.0 | CV research & multi-class evaluation |
| **Open Images Commercial Airliners** | Hugging Face: `a2015003713/military-aircraft-detection-dataset` | CC-BY 2.0 / CC-BY 4.0 | Academic and research benchmarking |
| **Civilian Marine Backgrounds** | Local Project Marine Collection | CC0 / Public Domain | Background negative false-positive rate evaluation |
| **Controlled Synthetic Fixtures** | Local OpenCV deterministic generation | CC0 / Public Domain | Codec, container, and edge-case testing |

---

## 3. Test Artifact Inventory

### A. Labeled Target Images (`tests/artifacts/images/`)
Total images: `30`
Annotations format: Standard YOLO (`class_id x_center y_center width height`) with normalized coordinates.

| Filename | Category | Resolution | Source | License | SHA-256 (First 12) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `rigo_rigoleto_train0022.jpg` | **drone** | 587x285 | Rigoleto/drone-detection | CC-BY 4.0 | `adb39cf64942...` |
| `rigo_rigoleto_train0034.jpg` | **drone** | 800x350 | Rigoleto/drone-detection | CC-BY 4.0 | `8eb3cbc2d5eb...` |
| `rigo_rigoleto_train0044.jpg` | **drone** | 4000x2850 | Rigoleto/drone-detection | CC-BY 4.0 | `d717fbb3670c...` |
| `rigo_rigoleto_train0053.jpg` | **drone** | 870x580 | Rigoleto/drone-detection | CC-BY 4.0 | `d01e8d79e36b...` |
| `rigo_rigoleto_train0062.jpg` | **drone** | 1200x800 | Rigoleto/drone-detection | CC-BY 4.0 | `cf13a87045ab...` |
| `rigo_rigoleto_train0069.jpg` | **drone** | 570x380 | Rigoleto/drone-detection | CC-BY 4.0 | `5bf98d914b15...` |
| `rigo_rigoleto_train0082.jpg` | **drone** | 1920x1282 | Rigoleto/drone-detection | CC-BY 4.0 | `3718201708d3...` |
| `vgt_quadcopter_drone_005.jpg` | **drone** | 640x480 | Authentic Drone Flight Capture | CC-BY 4.0 | `3c28d981d9d2...` |
| `vgt_quadcopter_drone_030.jpg` | **drone** | 640x480 | Authentic Drone Flight Capture | CC-BY 4.0 | `5d8bb80e0597...` |
| `vgt_quadcopter_drone_040.jpg` | **drone** | 640x480 | Authentic Drone Flight Capture | CC-BY 4.0 | `e57a05c2f886...` |
| `vgt_quadcopter_drone_080.jpg` | **drone** | 640x480 | Authentic Drone Flight Capture | CC-BY 4.0 | `e500083b4c3b...` |
| `vgt_quadcopter_drone_115.jpg` | **drone** | 640x480 | Authentic Drone Flight Capture | CC-BY 4.0 | `2106436d63b3...` |
| `air_airliner_022dae44ddec4c97e21bfeb082199914.jpg` | **airplane** | 1800x1196 | Open Images / a2015003713 | CC-BY 2.0 | `9cf2bd9f4f87...` |
| `air_airliner_04631f3ef35600b350580473753dc170.jpg` | **airplane** | 539x364 | Open Images / a2015003713 | CC-BY 2.0 | `b2979bcbb7b4...` |
| `air_airliner_062f90e6b178f49bd830e8a6dc7d4c93.jpg` | **airplane** | 1280x1358 | Open Images / a2015003713 | CC-BY 2.0 | `1e39a6509345...` |
| `air_airliner_06633fc1bc242e49d9c67e2d5550fd91.jpg` | **airplane** | 2048x1365 | Open Images / a2015003713 | CC-BY 2.0 | `efe64339c7a6...` |
| `air_airliner_07f23f94467aa125f85ecf6f23715ec8.jpg` | **airplane** | 2048x1253 | Open Images / a2015003713 | CC-BY 2.0 | `397d5be35c88...` |
| `air_airliner_081aa93a4d74ff1b9455de0759929364.jpg` | **airplane** | 1143x1600 | Open Images / a2015003713 | CC-BY 2.0 | `73befbdc101a...` |
| `air_airliner_09442d4599b9ddcfb54534187412059c.jpg` | **airplane** | 1280x841 | Open Images / a2015003713 | CC-BY 2.0 | `1e1c0c2b15e9...` |
| `air_airliner_0cfc60dc4963e658356b4b385dcfe409.jpg` | **airplane** | 1200x799 | Open Images / a2015003713 | CC-BY 2.0 | `381eabcd254d...` |
| `matis_0162_jpg.rf.d164b8e0f2e90d953b68536543c815a1.jpg` | **bird** | 640x640 | matisdsp/drone-bird-detection | CC-BY 4.0 | `93f80f7e5cc3...` |
| `matis_2984_jpg.rf.1c57341375fc275d875dffd334b3e33b.jpg` | **bird** | 640x640 | matisdsp/drone-bird-detection | CC-BY 4.0 | `da7e23de0fc8...` |
| `matis_3072_jpg.rf.5f741238a4e52c8d23cd4dd27607a6bf.jpg` | **bird** | 640x640 | matisdsp/drone-bird-detection | CC-BY 4.0 | `4c08f3501531...` |
| `matis_Acadian_Flycatcher_0019_795592_jpg.rf.44407034e9b173d95a448ceef1241f59.jpg` | **bird** | 640x640 | matisdsp/drone-bird-detection | CC-BY 4.0 | `134b29afc6a8...` |
| `matis_BirdsFlying_2291_jpg.rf.d64bfdda4d12f8ee0b8818cea3f02894.jpg` | **bird** | 965x607 | matisdsp/drone-bird-detection | CC-BY 4.0 | `15451bb7a27b...` |
| `matis_Blue_Jay_0048_62433_jpg.rf.cbd82f81e2fe4abf6cc5a8feda6fed53.jpg` | **bird** | 640x640 | matisdsp/drone-bird-detection | CC-BY 4.0 | `bd5474e81923...` |
| `matis_Fish_Crow_0037_26071_jpg.rf.2d6bd733287c69eeaf01cbb72121c2ca.jpg` | **bird** | 640x640 | matisdsp/drone-bird-detection | CC-BY 4.0 | `8c33d8b15186...` |
| `matis_pic_064_jpg.rf.320be1465e2e630687723ce577430aef.jpg` | **bird** | 640x640 | matisdsp/drone-bird-detection | CC-BY 4.0 | `941897917f59...` |
| `balloon_aw_00030.jpg` | **balloon** | 640x360 | Civilian Balloon Dataset | CC0 / Public Domain | `cad30025595c...` |
| `balloon_aw_00072.jpg` | **balloon** | 640x360 | Civilian Balloon Dataset | CC0 / Public Domain | `2dcc6cc65dfd...` |

### B. Background Negative Images (`tests/artifacts/negative_cases/`)
Total negative images: `5`
All negative cases have verified empty ground-truth (`0` annotations).

| Filename | Visual Scene | Resolution | SHA-256 (First 12) |
| :--- | :--- | :--- | :--- |
| `neg_boat_deck_00.jpg` | Marine deck and horizon - zero aerial targets | 640x300 | `c6ae7d472ec3...` |
| `neg_boat_deck_12.jpg` | Ocean water and boat infrastructure | 640x300 | `4cfd9ca63a1f...` |
| `neg_boat_deck_24.jpg` | Open sea with calm water | 640x300 | `4b43ef3e44b6...` |
| `neg_boat_deck_36.jpg` | Harbor railing and overcast sky | 640x300 | `585ded89aec6...` |
| `neg_clear_sky_gradient.jpg` | Smooth synthetic sky gradient fixture (zero targets) | 640x480 | `b0a5cf915bfe...` |

### C. Video Fixtures (`tests/artifacts/videos/`)
Total test videos: `4`

| Filename | Format | Frames | FPS | Resolution | Target Content |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `civilian_drone_surveillance.mp4` | MP4 (H.264 / AVC) | 84 | 15.0 | 640x480 | Real civilian drone surveillance patrol flight in open airspace |
| `drone_vs_bird_perimeter.mp4` | MP4 (H.264 / AVC) | 110 | 25.0 | 640x480 | Civilian drone vs bird airspace perimeter discrimination test clip |
| `civilian_flight_mjpg.avi` | AVI (Motion-JPEG) | 60 | 30.0 | 640x360 | Deterministic AVI container test fixture for backend compatibility |
| `empty_sky_negative.mp4` | MP4 (H.264 / AVC) | 60 | 30.0 | 640x360 | Empty sky video with moving cloud formations (zero detections expected) |

### D. Corrupt & Malformed Edge-Case Fixtures (`tests/artifacts/corrupt_inputs/`)
Total edge-case files: `5`

| Filename | File Type / Defect | Size | Expected Behavior |
| :--- | :--- | :--- | :--- |
| `zero_byte_video.mp4` | Zero-byte MP4 | 0 bytes | Graceful rejection with clear error message |
| `zero_byte_image.jpg` | Zero-byte JPEG | 0 bytes | Graceful rejection with clear error message |
| `corrupt_header.mp4` | Corrupt MP4 Container | 2048 bytes | Graceful rejection with clear error message |
| `truncated_image.jpg` | Truncated JPEG | 14 bytes | Graceful rejection with clear error message |
| `unsupported_format.bin` | Raw Binary (.bin) | 50 bytes | Graceful rejection with clear error message |

---

## 4. Verification & Integrity Checklist

- [x] All images, videos, and negative fixtures exist on local disk.
- [x] All SHA-256 checksums are deterministically computed and verifiable.
- [x] Ground-truth bounding boxes are strictly normalized to civilian classes (0: drone, 1: airplane, 2: bird, 3: balloon).
- [x] Zero physical camera hardware is required for the automated execution of integration tests.
- [x] All media licenses have been verified and documented.
