# Zig Raytracer

A small raytracer built in [Zig](https://ziglang.org/).

![screenshot of the rendered scene](https://mq32.de/public/zig-tracer-01.png)

## Build Instructions

Just have `zig` in your path and type:
```
zig build-exe tracer.zig # build
./tracer                 # & run
```

The raytracer will output a [PGM](https://de.wikipedia.org/wiki/Portable_Anymap) file that can be displayed with
various image viewers.