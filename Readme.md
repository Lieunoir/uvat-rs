# UVAT-rs

Partial reimplementation in Rust of the paper ["Joint optimization of distortion and cut location for mesh parameterization using an Ambrosio-Tortorelli functional"](https://perso.liris.cnrs.fr/david.coeurjolly/publication/uv-at/uv-at.pdf) with better performances.

Only the variational part is provided: the final cutting part is missing, as well as the initialization part (which should be provided using Tutte's method after an initial cut).
