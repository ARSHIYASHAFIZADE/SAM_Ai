/** @type {import('next').NextConfig} */
const BACKEND = process.env.BACKEND_URL || "http://localhost:5000";

const nextConfig = {
  output: "standalone",
  devIndicators: false,
  async rewrites() {
    return [
      {
        source: "/api-backend/:path*",
        destination: `${BACKEND}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
