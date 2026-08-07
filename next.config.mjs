/**
 * Project: chromadb-embedding-visualizer
 * File:    next.config.mjs
 *
 * Description:
 * Next.js build configuration for the frontend.
 *
 * Author:
 * Jan Alexandr Kopřiva
 * jan.alexandr.kopriva@gmail.com
 *
 * License: MIT
 */

/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
 
}

export default nextConfig
