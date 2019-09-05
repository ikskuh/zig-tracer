const std = @import("std");
const warn = std.debug.warn;
const assert = std.debug.assert;

const File = std.fs.File;
const ASCII = std.ASCII;

const TypeId = @import("builtin").TypeId;
const TypeInfo = @import("builtin").TypeInfo;

fn clamp(comptime T: type, v: T, min: T, max: T) T {
    if (v < min) {
        return min;
    } else {
        if (v > max) {
            return max;
        } else {
            return v;
        }
    }
}

fn lerp(a: f32, b: f32, f: f32) f32 {
    return (1.0 - f) * a + f * b;
}

fn Color(comptime T: type) type {
    return packed struct {
        const Self = @This();
        const Component = T;

        const black = Self{ .R = 0, .G = 0, .B = 0 };
        const white = Self{ .R = 1, .G = 1, .B = 1 };

        R: Component,
        G: Component,
        B: Component,

        fn gray(v: T) Self {
            return Self{ .R = v, .G = v, .B = v };
        }

        fn rgb(r: T, g: T, b: T) Self {
            return Self{
                .R = r,
                .G = g,
                .B = b,
            };
        }

        fn equals(self: Self, b: Self) bool {
            return self.R == b.R and self.G == b.G and self.B == b.B;
        }

        fn mix(a: Self, b: Self, f: f32) Self {
            return Self{
                .R = lerp(a.R, b.R, f),
                .G = lerp(a.G, b.G, f),
                .B = lerp(a.B, b.B, f),
            };
        }

        fn applyGamma(a: Self, gamma: f32) Self {
            return Self{
                .R = std.math.pow(f32, a.R, gamma),
                .G = std.math.pow(f32, a.G, gamma),
                .B = std.math.pow(f32, a.B, gamma),
            };
        }

        fn toSRGB(self: Self) Self {
            return self.applyGamma(1.0 / 2.2);
        }

        fn toLinear(self: Self) Self {
            return self.applyGamma(2.2);
        }
    };
}

fn isInteger(comptime T: type) bool {
    return @typeInfo(T) == TypeId.Int;
}

fn isFloat(comptime T: type) bool {
    return @typeInfo(T) == TypeId.Float;
}

fn map_to_uniform(value: var) f64 {
    const T = @typeOf(value);
    switch (@typeInfo(T)) {
        TypeId.Int => |t| {
            if (t.is_signed) {
                const min = @intToFloat(f64, -(1 << (t.bits - 1)));
                const max = @intToFloat(f64, (1 << (t.bits - 1)));
                return (@intToFloat(f64, value) - min) / (max - min);
            } else {
                const max = @intToFloat(f64, (1 << t.bits) - 1);
                return @intToFloat(f64, value) / max;
            }
        },
        TypeId.Float => |t| {
            return f64(value);
        },
        else => @compileError(@typeName(@typeOf(value)) ++ " is neither integer nor float!"),
    }
}

fn map_from_uniform(comptime T: type, value: f64) T {
    switch (@typeInfo(T)) {
        TypeId.Int => |t| {
            if (t.is_signed) {
                const min = @intToFloat(f64, -(1 << (t.bits - 1)));
                const max = @intToFloat(f64, (1 << (t.bits - 1)));
                return @floatToInt(T, std.math.floor((max - min) * value + min));
            } else {
                const max = @intToFloat(f64, (1 << t.bits) - 1);
                return @floatToInt(T, std.math.floor(max * value));
            }
        },
        TypeId.Float => |t| {
            return @floatCast(T, value);
        },
        else => @panic("Unsupported type " ++ @typeName(T)),
    }
}

fn mapColor(comptime TOut: type, c: var) TOut {
    const TIn = @typeOf(c);
    if (TIn == TOut)
        return c;

    var r = map_to_uniform(c.R);
    var g = map_to_uniform(c.G);
    var b = map_to_uniform(c.B);

    r = clamp(f64, r, 0.0, 1.0);
    g = clamp(f64, g, 0.0, 1.0);
    b = clamp(f64, b, 0.0, 1.0);

    return TOut{
        .R = map_from_uniform(TOut.Component, r),
        .G = map_from_uniform(TOut.Component, g),
        .B = map_from_uniform(TOut.Component, b),
    };
}

test "mapColor" {
    comptime {
        var c_u8 = Color(u8){ .R = 0xFF, .G = 0x00, .B = 0x00 };
        var c_f32 = Color(f32){ .R = 1.0, .G = 0.0, .B = 0.0 };

        assert(mapColor(@typeOf(c_f32), c_u8).equals(c_f32));
        assert(mapColor(@typeOf(c_u8), c_f32).equals(c_u8));
    }
}

fn RGB(comptime hexspec: [7]u8) error{InvalidCharacter}!Color(u8) {
    if (hexspec[0] != '#')
        return error.InvalidCharacter;
    return Color(u8){
        .R = ((try std.fmt.charToDigit(hexspec[1], 16)) << 4) | (try std.fmt.charToDigit(hexspec[2], 16)),
        .G = ((try std.fmt.charToDigit(hexspec[3], 16)) << 4) | (try std.fmt.charToDigit(hexspec[4], 16)),
        .B = ((try std.fmt.charToDigit(hexspec[5], 16)) << 4) | (try std.fmt.charToDigit(hexspec[6], 16)),
    };
}

const RED = RGB("#FF0000");
const GREEN = RGB("#00FF00");
const BLUE = RGB("#0000FF");

test "RGB" {
    // assert(compareColors(try RGB("#123456"), Color { .R = 0x12, .G = 0x34, .B = 0x56 }));
    assert((try RGB("#123456")).equals(Color(u8){ .R = 0x12, .G = 0x34, .B = 0x56 }));
    assert(if (RGB("!000000")) |c| false else |err| err == error.InvalidCharacter);
    assert(if (RGB("#X00000")) |c| false else |err| err == error.InvalidCharacter);
}

fn Bitmap(comptime TColor: type, comptime width: comptime_int, comptime height: comptime_int) type {
    return struct {
        const Self = @This();
        const MyColor = TColor;

        pixels: [width * height]MyColor,

        fn create() Self {
            return Self{ .pixels = undefined };
        }

        fn get_width(_: Self) usize {
            return width;
        }
        fn get_height(_: Self) usize {
            return height;
        }

        fn get(self: Self, x: usize, y: usize) error{OutOfRange}!MyColor {
            if (x >= width or y >= height) {
                return error.OutOfRange;
            } else {
                return self.pixels[y * width + x];
            }
        }

        fn set(self: *Self, x: usize, y: usize, color: MyColor) error{OutOfRange}!void {
            if (x >= width or y >= height) {
                return error.OutOfRange;
            } else {
                self.pixels[y * width + x] = color;
            }
        }
    };
}

fn Times(comptime len: comptime_int) [len]void {
    const val: [len]void = undefined;
    return val;
}

const float = f32;

const Vec2 = struct {
    x: float,
    y: float,
};

const Vec3 = struct {
    const zero = Vec3{ .x = 0, .y = 0, .z = 0 };
    const up = Vec3{ .x = 0, .y = 1, .z = 0 };
    const forward = Vec3{ .x = 0, .y = 0, .z = -1 };
    const right = Vec3{ .x = 1, .y = 0, .z = 0 };

    x: float,
    y: float,
    z: float,

    fn length(self: Vec3) float {
        return std.math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z);
    }

    fn add(self: Vec3, other: Vec3) Vec3 {
        return Vec3{
            .x = self.x + other.x,
            .y = self.y + other.y,
            .z = self.z + other.z,
        };
    }

    fn sub(self: Vec3, other: Vec3) Vec3 {
        return Vec3{
            .x = self.x - other.x,
            .y = self.y - other.y,
            .z = self.z - other.z,
        };
    }

    fn scale(self: Vec3, f: float) Vec3 {
        return Vec3{
            .x = self.x * f,
            .y = self.y * f,
            .z = self.z * f,
        };
    }

    fn normalize(self: Vec3) Vec3 {
        return self.scale(1.0 / self.length());
    }

    fn dot(self: Vec3, other: Vec3) float {
        return self.x * other.x + self.y * other.y + self.z * other.z;
    }

    fn cross(a: Vec3, b: Vec3) Vec3 {
        return Vec3{
            .x = a.y * b.z - a.z * b.y,
            .y = a.z * b.x - a.x * b.z,
            .z = a.x * b.y - a.y * b.x,
        };
    }

    fn distance(a: Vec3, b: Vec3) float {
        return a.sub(b).length();
    }
};

fn vec3(x: float, y: float, z: float) Vec3 {
    return Vec3{
        .x = x,
        .y = y,
        .z = z,
    };
}

const Plane = struct {
    origin: Vec3,
    normal: Vec3,

    fn toGeometry(self: @This()) Geometry {
        return Geometry{ .plane = self };
    }
};

const Sphere = struct {
    origin: Vec3,
    radius: f32,

    fn toGeometry(self: @This()) Geometry {
        return Geometry{ .sphere = self };
    }
};

const Geometry = union(enum) {
    plane: Plane,
    sphere: Sphere,
};

const Material = struct {
    albedo: Color(f32),
    roughness: float,
    metalness: float,
};

const Object = struct {
    geometry: Geometry,
    material: Material,
};

const Ray = struct {
    origin: Vec3,
    direction: Vec3,

    const Hit = struct {
        position: Vec3,
        normal: Vec3,
        distance: float,
    };

    fn intersect(ray: Ray, obj: Geometry) ?Hit {
        switch (obj) {
            Geometry.plane => |plane| {
                const denom = plane.normal.dot(ray.direction);
                if (denom < -1e-6) {
                    const p0l0 = plane.origin.sub(ray.origin);
                    const t = p0l0.dot(plane.normal) / denom;
                    if (t >= 0) {
                        return Hit{
                            .distance = t,
                            .position = ray.origin.add(ray.direction.scale(t)),
                            .normal = plane.normal,
                        };
                    }
                }
                return null;
            },
            Geometry.sphere => |sphere| {

                // geometric solution
                const L = sphere.origin.sub(ray.origin);
                var tca = L.dot(ray.direction);
                // if (tca < 0) return false;
                var d2 = L.dot(L) - tca * tca;
                if (d2 > sphere.radius * sphere.radius) {
                    return null;
                }
                const thc = std.math.sqrt(sphere.radius * sphere.radius - d2);
                const t0 = tca - thc;
                const t1 = tca + thc;

                const d = std.math.min(t0, t1);
                if (d >= 0) {
                    return Hit{
                        .distance = d,
                        .position = ray.origin.add(ray.direction.scale(d)),
                        .normal = ray.origin.add(ray.direction.scale(d)).sub(sphere.origin).normalize(),
                    };
                }
            },
        }
        return null;
    }
};

const Camera = struct {
    origin: Vec3,
    forward: Vec3,
    up: Vec3,
    right: Vec3,
    focalLength: float,

    fn lookAt(origin: Vec3, target: Vec3, up: Vec3, focalLength: float) Camera {
        var cam = Camera{
            .origin = origin,
            .forward = target.sub(origin).normalize(),
            .up = undefined,
            .right = undefined,
            .focalLength = focalLength,
        };
        cam.right = up.cross(cam.forward).normalize();
        cam.up = cam.forward.cross(cam.right).normalize();
        return cam;
    }

    fn constructRay(self: Camera, uv: Vec2) Ray {
        return Ray{
            .origin = self.origin,
            .direction = self.forward.scale(self.focalLength).add(self.right.scale(uv.x)).add(self.up.scale(uv.y)).normalize(),
        };
    }
};

const scene = [_]Object{
    Object{
        .geometry = (Plane{
            .origin = Vec3.zero,
            .normal = Vec3.up,
        }).toGeometry(),
        .material = Material{
            .albedo = Color(f32).gray(0.8),
            .roughness = 1.0,
            .metalness = 0.0,
        },
    }, Object{
        .geometry = (Sphere{
            .origin = vec3(0, 0, 5),
            .radius = 1.0,
        }).toGeometry(),
        .material = Material{
            .albedo = Color(f32).rgb(1.0, 0.0, 0.0),
            .roughness = 1.0,
            .metalness = 0.0,
        },
    }, Object{
        .geometry = (Sphere{
            .origin = vec3(5, 0, 5),
            .radius = 1.0,
        }).toGeometry(),
        .material = Material{
            .albedo = Color(f32).rgb(0.0, 1.0, 0.0),
            .roughness = 1.0,
            .metalness = 0.0,
        },
    }, Object{
        .geometry = (Sphere{
            .origin = vec3(5, 4, 5),
            .radius = 1.0,
        }).toGeometry(),
        .material = Material{
            .albedo = Color(f32).rgb(0.0, 0.0, 1.0),
            .roughness = 1.0,
            .metalness = 0.0,
        },
    },
};

const camera = Camera.lookAt(vec3(0, 2, 0), vec3(0, 2, 100), Vec3.up, 1.0);

const HitTestResult = struct {
    geom: Ray.Hit,
    obj: *const Object,
};

fn hitTest(ray: Ray) ?HitTestResult {
    var hit: ?HitTestResult = null;
    for (scene) |*obj| {
        if (ray.intersect(obj.geometry)) |pt| {
            const hit_obj = HitTestResult{
                .geom = pt,
                .obj = obj,
            };

            if (hit) |*h2| {
                if (h2.geom.distance > pt.distance) {
                    h2.* = hit_obj;
                }
            } else {
                hit = hit_obj;
            }
        }
    }
    return hit;
}

fn render(uv: Vec2) Color(f32) {
    const ray = camera.constructRay(uv);

    const maybe_hit = hitTest(ray);

    if (maybe_hit) |hit| {
        var col = hit.obj.material.albedo;

        if (hitTest(Ray{
            .origin = hit.geom.position.add(hit.geom.normal.scale(1e-5)),
            .direction = vec3(0.7, 1, -0.5).normalize(),
        })) |_| {
            col.R *= 0.5;
            col.G *= 0.5;
            col.B *= 0.5;
        }

        return col.toSRGB();
    } else {
        return Color(f32){ .R = 0.8, .G = 0.8, .B = 1.0 };
    }
}

var ms_rng = std.rand.DefaultPrng.init(0);

fn render_multisample(uv: Vec2, sampleCount: usize, sampleRadius: Vec2) Color(f32) {
    var result = Color(f32).black;
    if (sampleCount <= 1) {
        result = render(uv);
    } else {
        var i: usize = 0;
        while (i < sampleCount) : (i += 1) {
            var dx = ms_rng.random.float(f32);
            var dy = ms_rng.random.float(f32);
            var c = render(Vec2{
                .x = uv.x + 0.5 * sampleRadius.x * (dx - 0.5),
                .y = uv.y + 0.5 * sampleRadius.y * (dy - 0.5),
            });
            result.R += c.R;
            result.G += c.G;
            result.B += c.B;
        }
    }
    return Color(f32).mix(Color(f32).black, result, 1.0 / @intToFloat(f32, sampleCount));
}

const pic = struct {
    var bmp = Bitmap(Color(u8), 320, 240).create();
};

pub fn main() !void {
    var bmp = &pic.bmp;

    const image_size = Vec2{
        .x = @intToFloat(float, bmp.get_width()),
        .y = @intToFloat(float, bmp.get_height()),
    };

    const aspect: float = image_size.x / image_size.y;
    const pixel_scale = Vec2{
        .x = 1.0 / (image_size.x - 1),
        .y = 1.0 / (image_size.y - 1),
    };

    var y: usize = 0;
    while (y < bmp.get_height()) : (y += 1) {
        var x: usize = 0;
        while (x < bmp.get_width()) : (x += 1) {
            const uv = Vec2{
                .x = aspect * (2.0 * @intToFloat(float, x) * pixel_scale.x - 1.0),
                .y = 1.0 - 2.0 * @intToFloat(float, y) * pixel_scale.y,
            };

            const val = render_multisample(uv, 1, Vec2{ .x = aspect * pixel_scale.x, .y = pixel_scale.y });

            const mapped = mapColor(Color(u8), val);

            try bmp.set(x, y, mapped);
        }
        if ((y % 10) == 9) {
            warn("\rrender process: {} %  ", @floatToInt(i32, std.math.ceil(100.0 * @intToFloat(f32, y + 1) / image_size.y)));
        }
    }
    warn("\ndone!\n");

    try savePGM(bmp, "result.pgm");
}

fn savePGM(bmp: var, file: []const u8) !void {
    assert(@typeOf(bmp.*).MyColor.Component == u8);

    var f = try File.openWrite("result.pgm");
    defer f.close();

    var buf: [64]u8 = undefined;
    try f.write(try std.fmt.bufPrint(buf[0..], "P6 {} {} 255\n", bmp.get_width(), bmp.get_height()));

    try f.write(@sliceToBytes(bmp.pixels[0..]));
}
