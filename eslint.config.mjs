/**
 * Project: chromadb-embedding-visualizer
 * File:    eslint.config.mjs
 *
 * Description:
 * ESLint configuration for the frontend.
 *
 * Author:
 * Jan Alexandr Kopřiva
 * jan.alexandr.kopriva@gmail.com
 *
 * License: MIT
 */

import nextCoreWebVitals from "eslint-config-next/core-web-vitals"
import nextTypescript from "eslint-config-next/typescript"

export default [
  {
    ignores: [".next/**", "node_modules/**", ".venv/**", "data/**", "backend/**", "public/**"],
  },
  ...nextCoreWebVitals,
  ...nextTypescript,
]
