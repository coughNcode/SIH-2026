// DepthWizard backend — C++ HTTP server
//
// Serves the static frontend from ./public/ and provides:
//   POST /api/process  — image decode/resize + proxy to Python inference
//   GET  /api/metrics  — proxy to Python /metrics endpoint
//   GET  /api/health   — checks Python API health
//
// The real HeightNet inference runs in the Python FastAPI server (api/server.py).
// This C++ server handles:
//   1) Static file serving (Three.js frontend)
//   2) Image decode, resize to 224x224, re-encode as base64 for Python
//   3) Fallback height proxy if Python is unreachable (for offline demo)
//
// Build (MSVC via Developer Command Prompt):
//   cl /O2 /std:c++17 /EHsc /I include main.cpp /link ws2_32.lib
// Build (MinGW/g++):
//   g++ -O2 -std=c++17 -Iinclude main.cpp -o depthwizard_server -lws2_32 -lpthread
// Run:
//   depthwizard_server.exe 8080

#define CPPHTTPLIB_NO_EXCEPTIONS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "include/httplib.h"
#include "include/stb_image.h"
#include "include/stb_image_write.h"
#include "include/stb_image_resize2.h"

#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <algorithm>

// Python API address (local)
static const char* PYTHON_HOST = "localhost";
static const int   PYTHON_PORT = 8000;

// Working grid for image + heightmap
constexpr int GRID = 224;

// ─────────────────────────────────────────────────────────────────────────────
// Base64 encoder
// ─────────────────────────────────────────────────────────────────────────────
namespace b64 {
static const char* TABLE =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string encode(const unsigned char* data, size_t len) {
  std::string out;
  out.reserve(((len + 2) / 3) * 4);
  size_t i = 0;
  while (i + 3 <= len) {
    unsigned int n = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
    out += TABLE[(n >> 18) & 0x3F];
    out += TABLE[(n >> 12) & 0x3F];
    out += TABLE[(n >> 6)  & 0x3F];
    out += TABLE[n         & 0x3F];
    i += 3;
  }
  size_t rem = len - i;
  if (rem == 1) {
    unsigned int n = data[i] << 16;
    out += TABLE[(n >> 18) & 0x3F];
    out += TABLE[(n >> 12) & 0x3F];
    out += "==";
  } else if (rem == 2) {
    unsigned int n = (data[i] << 16) | (data[i + 1] << 8);
    out += TABLE[(n >> 18) & 0x3F];
    out += TABLE[(n >> 12) & 0x3F];
    out += TABLE[(n >> 6)  & 0x3F];
    out += "=";
  }
  return out;
}
}  // namespace b64

// ─────────────────────────────────────────────────────────────────────────────
// In-memory PNG encode
// ─────────────────────────────────────────────────────────────────────────────
struct MemBuf { std::vector<unsigned char> data; };
static void mem_write_cb(void* ctx, void* data, int size) {
  MemBuf* buf = reinterpret_cast<MemBuf*>(ctx);
  const unsigned char* p = reinterpret_cast<const unsigned char*>(data);
  buf->data.insert(buf->data.end(), p, p + size);
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian blur (separable)
// ─────────────────────────────────────────────────────────────────────────────
void gaussian_blur(std::vector<float>& img, int w, int h, int radius) {
  if (radius <= 0) return;
  std::vector<float> kernel(2 * radius + 1);
  float sigma = radius / 2.0f + 0.5f, sum = 0.0f;
  for (int i = -radius; i <= radius; ++i) {
    float v = std::exp(-(i * i) / (2.0f * sigma * sigma));
    kernel[i + radius] = v; sum += v;
  }
  for (auto& v : kernel) v /= sum;

  std::vector<float> tmp(img.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float acc = 0.0f;
      for (int k = -radius; k <= radius; ++k)
        acc += img[y * w + std::clamp(x + k, 0, w - 1)] * kernel[k + radius];
      tmp[y * w + x] = acc;
    }
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float acc = 0.0f;
      for (int k = -radius; k <= radius; ++k)
        acc += tmp[std::clamp(y + k, 0, h - 1) * w + x] * kernel[k + radius];
      img[y * w + x] = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fallback height proxy (used when Python API is unreachable)
// Returns a [0,1] float heightmap from RGB luminance + gradient.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> generate_height_proxy(const unsigned char* rgb, int w, int h) {
  std::vector<float> lum(w * h);
  for (int i = 0; i < w * h; ++i) {
    const unsigned char* px = rgb + i * 3;
    lum[i] = 0.2126f * px[0] / 255.0f
           + 0.7152f * px[1] / 255.0f
           + 0.0722f * px[2] / 255.0f;
  }
  std::vector<float> ht(w * h, 0.0f);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      int xm = std::max(x - 1, 0), xp = std::min(x + 1, w - 1);
      int ym = std::max(y - 1, 0), yp = std::min(y + 1, h - 1);
      float gx = lum[y * w + xp] - lum[y * w + xm];
      float gy = lum[yp * w + x] - lum[ym * w + x];
      float grad = std::sqrt(gx * gx + gy * gy);
      ht[y * w + x] = 0.7f * lum[y * w + x] + 0.3f * std::min(grad * 4.0f, 1.0f);
    }
  gaussian_blur(ht, w, h, 3);
  float mn = *std::min_element(ht.begin(), ht.end());
  float mx = *std::max_element(ht.begin(), ht.end());
  float rng = std::max(mx - mn, 1e-6f);
  for (auto& v : ht) v = (v - mn) / rng;
  return ht;
}

// ─────────────────────────────────────────────────────────────────────────────
// Try to call the Python inference API.
// Returns true and fills out_json on success; returns false on failure.
// ─────────────────────────────────────────────────────────────────────────────
bool try_python_inference(const std::string& image_b64, std::string& out_json) {
  httplib::Client py(PYTHON_HOST, PYTHON_PORT);
  py.set_connection_timeout(2, 0);
  py.set_read_timeout(30, 0);

  std::string body = "{\"image_b64\":\"" + image_b64 + "\"}";
  auto res = py.Post("/infer", body, "application/json");
  if (!res || res->status != 200) return false;
  out_json = res->body;
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler: POST /api/process
// ─────────────────────────────────────────────────────────────────────────────
void handle_process(const httplib::Request& req, httplib::Response& res) {
  if (!req.form.has_file("image")) {
    res.status = 400;
    res.set_content("{\"error\":\"missing 'image' field\"}", "application/json");
    return;
  }
  const auto& file = req.form.get_file("image");

  // Decode incoming image
  int w, h, ch;
  unsigned char* decoded = stbi_load_from_memory(
      reinterpret_cast<const unsigned char*>(file.content.data()),
      static_cast<int>(file.content.size()), &w, &h, &ch, 3);
  if (!decoded) {
    res.status = 400;
    res.set_content("{\"error\":\"could not decode image\"}", "application/json");
    return;
  }

  // Resize to GRID x GRID
  std::vector<unsigned char> resized(GRID * GRID * 3);
  stbir_resize_uint8_linear(decoded, w, h, 0, resized.data(), GRID, GRID, 0, STBIR_RGB);
  stbi_image_free(decoded);

  // Encode resized RGB as PNG then base64
  MemBuf rgb_png;
  stbi_write_png_to_func(mem_write_cb, &rgb_png, GRID, GRID, 3, resized.data(), GRID * 3);
  std::string rgb_b64 = b64::encode(rgb_png.data.data(), rgb_png.data.size());

  // Also encode resized image as data URL for response
  std::string image_data_url = "data:image/png;base64," + rgb_b64;

  // ── Try Python API first ────────────────────────────────────────────────
  std::string py_json;
  if (try_python_inference(rgb_b64, py_json)) {
    // Python returned its own JSON; relay it back directly, augmented with image
    // Strip closing brace and append the image field
    if (!py_json.empty() && py_json.back() == '}') {
      py_json.pop_back();
      py_json += ",\"image\":\"" + image_data_url + "\"}";
    }
    res.set_content(py_json, "application/json");
    return;
  }

  // ── Fallback: local C++ height proxy ────────────────────────────────────
  std::cerr << "[warn] Python API unreachable — using C++ fallback proxy\n";

  std::vector<float> height = generate_height_proxy(resized.data(), GRID, GRID);

  std::vector<unsigned char> h_u8(GRID * GRID);
  for (int i = 0; i < GRID * GRID; ++i)
    h_u8[i] = static_cast<unsigned char>(std::clamp(height[i] * 255.0f, 0.0f, 255.0f));

  MemBuf hm_png;
  stbi_write_png_to_func(mem_write_cb, &hm_png, GRID, GRID, 1, h_u8.data(), GRID);
  std::string hm_b64 = b64::encode(hm_png.data.data(), hm_png.data.size());

  std::ostringstream json;
  json << "{"
       << "\"width\":"       << GRID  << ","
       << "\"height\":"      << GRID  << ","
       << "\"pred_min_m\":"  << 0.0   << ","
       << "\"pred_max_m\":"  << 30.0  << ","
       << "\"pred_mean_m\":" << 5.0   << ","
       << "\"source\":\"C++ fallback (Python offline)\","
       << "\"image\":\"data:image/png;base64," << rgb_b64 << "\","
       << "\"heightmap\":\"data:image/png;base64," << hm_b64 << "\""
       << "}";
  res.set_content(json.str(), "application/json");
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler: GET /api/metrics  — proxy to Python
// ─────────────────────────────────────────────────────────────────────────────
void handle_metrics(const httplib::Request& req, httplib::Response& res) {
  (void)req;
  httplib::Client py(PYTHON_HOST, PYTHON_PORT);
  py.set_connection_timeout(2, 0);
  py.set_read_timeout(5, 0);
  auto pyres = py.Get("/metrics");
  if (!pyres || pyres->status != 200) {
    res.status = 503;
    res.set_content("{\"error\":\"Python API unavailable — run api/server.py\"}", "application/json");
    return;
  }
  res.set_content(pyres->body, "application/json");
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler: GET /api/health
// ─────────────────────────────────────────────────────────────────────────────
void handle_health(const httplib::Request& req, httplib::Response& res) {
  (void)req;
  httplib::Client py(PYTHON_HOST, PYTHON_PORT);
  py.set_connection_timeout(1, 0);
  auto pyres = py.Get("/health");
  std::string status = (pyres && pyres->status == 200) ? "python_ok" : "python_offline";
  res.set_content("{\"cpp\":\"ok\",\"python\":\"" + status + "\"}", "application/json");
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
  int port = 8080;
  if (argc > 1) port = std::atoi(argv[1]);

  httplib::Server server;

  // Serve frontend static files
  server.set_mount_point("/", "./public");

  // API routes
  server.Post("/api/process", handle_process);
  server.Get ("/api/metrics", handle_metrics);
  server.Get ("/api/health",  handle_health);

  // CORS
  server.set_default_headers({
    {"Access-Control-Allow-Origin",  "*"},
    {"Access-Control-Allow-Headers", "*"},
    {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
  });
  server.Options(R"(/.*)", [](const httplib::Request&, httplib::Response& r) {
    r.status = 200;
  });

  std::cout << "DepthWizard C++ backend  http://localhost:" << port << "\n";
  std::cout << "Python inference API     http://localhost:" << PYTHON_PORT << "\n";
  std::cout << "Serving frontend from    ./public/\n";
  server.listen("0.0.0.0", port);
  return 0;
}