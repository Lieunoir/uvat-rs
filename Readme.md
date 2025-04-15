# UVAT-rs

| ![screenshot_000](https://github.com/user-attachments/assets/1275af70-a28a-4717-9051-4b0dc699bedb) | ![screenshot_001](https://github.com/user-attachments/assets/af61b0e6-d595-42d0-83ad-4162d619139c) |
| - | - |
| Resulting distorsion | Parameterization with $v$ |

Partial Rust reimplementation of the paper ["Joint optimization of distortion and cut location for mesh parameterization using an Ambrosio-Tortorelli functional"](https://perso.liris.cnrs.fr/david.coeurjolly/publication/uv-at/uv-at.pdf) with better performances.

Only the variational part is provided: the final cutting part is missing, as well as the initialization part (which should be provided using Tutte's method after an initial cut).

An interactive demo can be run using `cargo run -r --example demo`. Obj files to use can be found [here](https://github.com/Lieunoir/UV-AT/tree/main/input).
