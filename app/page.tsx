"use client"

import type React from "react"
import { useState, useRef, useEffect, useCallback, useMemo } from "react"
import { Canvas, useThree, useFrame } from "@react-three/fiber"
import { OrbitControls } from "@react-three/drei"
import * as THREE from "three"
import { Button } from "@/components/ui/button"
import { Sun, Moon, ChevronDown, ChevronUp } from "lucide-react"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

type EmbeddingPoint = {
  id: string
  text: string
  x: number
  y: number
  z: number
  cluster?: number
  topic_name?: string
  metadata?: Record<string, unknown>
}

type TopicInfo = {
  id: number
  name: string
  keywords: string[]
  count: number
}

type CollectionInfo = {
  name: string
  count: number
  dimensions: number
  model: string
  topics?: TopicInfo[]
}

// Color palette for clusters (40 distinct colors)
const CLUSTER_PALETTE = [
  [0.89, 0.10, 0.11], // red
  [0.22, 0.49, 0.72], // blue
  [0.30, 0.69, 0.29], // green
  [0.60, 0.31, 0.64], // purple
  [1.00, 0.50, 0.00], // orange
  [1.00, 1.00, 0.20], // yellow
  [0.65, 0.34, 0.16], // brown
  [0.97, 0.51, 0.75], // pink
  [0.50, 0.50, 0.50], // gray
  [0.74, 0.74, 0.13], // olive
  [0.09, 0.75, 0.81], // cyan
  [0.83, 0.15, 0.48], // magenta
  [0.10, 0.44, 0.39], // teal
  [0.55, 0.63, 0.80], // light blue
  [0.90, 0.57, 0.67], // salmon
  [0.40, 0.18, 0.57], // indigo
  [0.90, 0.73, 0.64], // peach
  [0.36, 0.36, 0.36], // dark gray
  [0.55, 0.71, 0.00], // lime
  [0.75, 0.00, 0.00], // dark red
  // Additional colors for more topics
  [0.00, 0.50, 0.50], // dark cyan
  [0.80, 0.40, 0.00], // burnt orange
  [0.50, 0.00, 0.50], // dark purple
  [0.00, 0.60, 0.30], // forest green
  [0.70, 0.70, 0.00], // dark yellow
  [0.60, 0.20, 0.40], // plum
  [0.20, 0.60, 0.60], // sea green
  [0.90, 0.30, 0.30], // coral
  [0.30, 0.30, 0.70], // slate blue
  [0.50, 0.70, 0.30], // yellow green
  [0.80, 0.20, 0.60], // rose
  [0.40, 0.50, 0.60], // steel blue
  [0.70, 0.50, 0.30], // tan
  [0.20, 0.40, 0.20], // dark green
  [0.60, 0.40, 0.60], // lavender
  [0.80, 0.60, 0.40], // sandy
  [0.30, 0.50, 0.80], // cornflower
  [0.70, 0.30, 0.50], // berry
  [0.40, 0.70, 0.50], // mint
  [0.90, 0.40, 0.10], // pumpkin
]

const CLUSTER_HEX = CLUSTER_PALETTE.map(c => 
  '#' + c.map(v => Math.round(v * 255).toString(16).padStart(2, '0')).join('')
)

// Instanced points with hover detection
function InstancedPoints({ 
  points, 
  onHover,
  selectedTopic
}: { 
  points: EmbeddingPoint[]
  onHover: (index: number | null, position: {x: number, y: number} | null) => void
  selectedTopic: number | null
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const { camera, raycaster, pointer, gl } = useThree()
  const lastHovered = useRef<number | null>(null)
  
  // Configure raycaster for small objects
  useEffect(() => {
    raycaster.params.Line = { threshold: 0.1 }
  }, [raycaster])
  
  // Don't render if no points
  const pointCount = points.length || 1
  
  const colorArray = useMemo(() => {
    if (points.length === 0) return new Float32Array(3)
    const colors = new Float32Array(points.length * 3)
    for (let i = 0; i < points.length; i++) {
      const cluster = points[i].cluster ?? 0
      const colorIdx = cluster === -1 ? 8 : Math.abs(cluster) % CLUSTER_PALETTE.length
      const color = CLUSTER_PALETTE[colorIdx]
      
      // If a topic is selected, dim non-matching points
      const isSelected = selectedTopic === null || cluster === selectedTopic
      const dimFactor = isSelected ? 1.0 : 0.15
      
      colors[i * 3] = color[0] * dimFactor
      colors[i * 3 + 1] = color[1] * dimFactor
      colors[i * 3 + 2] = color[2] * dimFactor
    }
    return colors
  }, [points, selectedTopic])

  useEffect(() => {
    if (!meshRef.current || points.length === 0) return
    
    const mesh = meshRef.current
    const dummy = new THREE.Object3D()
    
    for (let i = 0; i < points.length; i++) {
      const p = points[i]
      dummy.position.set(p.x * 2, p.y * 2, p.z * 2)
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
    }
    
    mesh.instanceMatrix.needsUpdate = true
  }, [points])

  // Update colors when selection changes
  useEffect(() => {
    if (!meshRef.current || points.length === 0) return
    const geometry = meshRef.current.geometry
    const colorAttr = geometry.getAttribute('color') as THREE.BufferAttribute | null
    if (colorAttr && colorAttr.array.length === colorArray.length) {
      for (let i = 0; i < colorArray.length; i++) {
        (colorAttr.array as Float32Array)[i] = colorArray[i]
      }
      colorAttr.needsUpdate = true
    }
  }, [colorArray, points.length, selectedTopic])

  // Raycast on every frame for hover detection
  useFrame(() => {
    if (!meshRef.current || points.length === 0) return
    
    raycaster.setFromCamera(pointer, camera)
    const intersects = raycaster.intersectObject(meshRef.current)
    
    if (intersects.length > 0 && intersects[0].instanceId !== undefined) {
      const idx = intersects[0].instanceId
      if (lastHovered.current !== idx) {
        lastHovered.current = idx
        // Get screen position
        const point = points[idx]
        if (point) {
          const vec = new THREE.Vector3(point.x * 2, point.y * 2, point.z * 2)
          vec.project(camera)
          const x = (vec.x * 0.5 + 0.5) * gl.domElement.clientWidth
          const y = (-vec.y * 0.5 + 0.5) * gl.domElement.clientHeight
          onHover(idx, { x, y })
        }
      }
    } else {
      if (lastHovered.current !== null) {
        lastHovered.current = null
        onHover(null, null)
      }
    }
  })

  if (points.length === 0) return null

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, pointCount]}>
      <sphereGeometry args={[0.035, 8, 8]}>
        <instancedBufferAttribute
          attach="attributes-color"
          args={[colorArray, 3]}
        />
      </sphereGeometry>
      <meshStandardMaterial vertexColors transparent opacity={0.85} roughness={0.4} metalness={0.1} />
    </instancedMesh>
  )
}

function Scene({ 
  points,
  onHover,
  darkMode,
  selectedTopic
}: { 
  points: EmbeddingPoint[]
  onHover: (index: number | null, position: {x: number, y: number} | null) => void
  darkMode: boolean
  selectedTopic: number | null
}) {
  return (
    <>
      <ambientLight intensity={darkMode ? 0.8 : 1.0} />
      <pointLight position={[0, 10, 0]} intensity={darkMode ? 0.5 : 0.6} />
      <gridHelper 
        args={[6, 30, darkMode ? "#333333" : "#d1d5db", darkMode ? "#222222" : "#e5e7eb"]} 
        rotation={[0, 0, 0]} 
      />
      <InstancedPoints points={points} onHover={onHover} selectedTopic={selectedTopic} />
      <OrbitControls 
        enableDamping 
        dampingFactor={0.05}
        minDistance={0.5}
        maxDistance={25}
      />
    </>
  )
}

export default function EmbeddingsVisualizer() {
  const [points, setPoints] = useState<EmbeddingPoint[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [loadingStatus, setLoadingStatus] = useState("")
  const [collectionInfo, setCollectionInfo] = useState<CollectionInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [darkMode, setDarkMode] = useState(true)
  
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)
  const [tooltipPos, setTooltipPos] = useState<{x: number, y: number} | null>(null)
  const [legendExpanded, setLegendExpanded] = useState(false)
  const [selectedTopic, setSelectedTopic] = useState<number | null>(null)
  const [windowSize, setWindowSize] = useState({ width: 1920, height: 1080 })

  // Update window size on client
  useEffect(() => {
    const updateSize = () => setWindowSize({ width: window.innerWidth, height: window.innerHeight })
    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  useEffect(() => {
    loadCollectionInfo()
  }, [])

  // Auto-load data when collection info is available
  useEffect(() => {
    if (collectionInfo && points.length === 0 && !isLoading) {
      loadData()
    }
  }, [collectionInfo])

  const loadCollectionInfo = async () => {
    try {
      const response = await fetch(`${API_URL}/api/info`)
      if (!response.ok) throw new Error("Failed to load collection info")
      const info = await response.json()
      setCollectionInfo(info)
      setError(null)
    } catch {
      setError("Nelze se připojit k backendu.")
    }
  }

  const loadData = async () => {
    if (!collectionInfo) return
    
    setIsLoading(true)
    setLoadingStatus(`Načítám ${collectionInfo.count.toLocaleString()} bodů...`)
    setError(null)
    
    try {
      const response = await fetch(`${API_URL}/api/embeddings?limit=${collectionInfo.count}`)
      if (!response.ok) throw new Error("Failed to load data")
      const data = await response.json()
      setPoints(data)
      setLoadingStatus("")
    } catch {
      setError("Chyba při načítání dat")
    } finally {
      setIsLoading(false)
    }
  }

  const handleHover = useCallback((index: number | null, position: {x: number, y: number} | null) => {
    setHoveredIndex(index)
    setTooltipPos(position)
  }, [])

  const themeClasses = darkMode 
    ? "bg-zinc-950 text-white" 
    : "bg-slate-100 text-slate-900"

  const hoveredPoint = hoveredIndex !== null ? points[hoveredIndex] : null

  // Main visualization (always shown)
  return (
    <main className={`${themeClasses} h-dvh flex flex-col overflow-hidden`}>
      {/* 3D Canvas */}
      <div className="flex-1 min-h-0 relative">
        {/* Loading overlay */}
        {(isLoading || points.length === 0) && (
          <div className={`absolute inset-0 z-40 flex items-center justify-center ${darkMode ? 'bg-zinc-950' : 'bg-gradient-to-br from-slate-100 to-slate-200'}`}>
            <div className="text-center">
              <div className={`animate-spin w-10 h-10 border-3 ${darkMode ? 'border-white' : 'border-slate-600'} border-t-transparent rounded-full mx-auto`} />
              <p className={`${darkMode ? 'text-zinc-400' : 'text-slate-600'} mt-4 font-mono text-sm`}>
                {error ? error : loadingStatus || "Načítám data..."}
              </p>
              {error && (
                <Button onClick={loadCollectionInfo} variant="outline" size="sm" className="mt-4">
                  Zkusit znovu
                </Button>
              )}
            </div>
          </div>
        )}

        <Canvas 
          camera={{ position: [5, 4, 5], fov: 50, near: 0.01, far: 1000 }}
          style={{ background: darkMode ? '#09090b' : '#f1f5f9' }}
        >
          <Scene points={points} onHover={handleHover} darkMode={darkMode} selectedTopic={selectedTopic} />
        </Canvas>
        
        {/* Info card - top left */}
        <div className={`absolute top-4 left-4 z-30 ${darkMode ? 'bg-zinc-900/90' : 'bg-white/95 border border-slate-200'} backdrop-blur rounded-lg shadow-lg p-3 text-xs`}>
          <div className={`font-bold text-sm mb-2 ${darkMode ? 'text-white' : 'text-slate-800'}`}>Embedding Visualizer</div>
          <div className="space-y-1">
            <div className="flex justify-between gap-4">
              <span className={darkMode ? 'text-zinc-400' : 'text-slate-500'}>Kolekce:</span>
              <span className={`font-mono ${darkMode ? 'text-white' : 'text-slate-700'}`}>{collectionInfo?.name || '-'}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className={darkMode ? 'text-zinc-400' : 'text-slate-500'}>Model:</span>
              <span className={`font-mono text-[10px] ${darkMode ? 'text-white' : 'text-slate-700'}`}>{collectionInfo?.model || '-'}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className={darkMode ? 'text-zinc-400' : 'text-slate-500'}>Záznamů:</span>
              <span className={`font-mono ${darkMode ? 'text-white' : 'text-slate-700'}`}>{points.length.toLocaleString()}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className={darkMode ? 'text-zinc-400' : 'text-slate-500'}>Dimenze:</span>
              <span className={`font-mono ${darkMode ? 'text-white' : 'text-slate-700'}`}>{collectionInfo?.dimensions || '-'}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className={darkMode ? 'text-zinc-400' : 'text-slate-500'}>Zdroj:</span>
              <span className={`font-mono ${darkMode ? 'text-white' : 'text-slate-700'}`}>Protext</span>
            </div>
          </div>
        </div>
        
        {/* Topic legend - left side, collapsible */}
        {collectionInfo?.topics && collectionInfo.topics.length > 0 && (
          <div 
            className={`absolute top-[175px] left-4 z-30 ${darkMode ? 'bg-zinc-900/90' : 'bg-white/95 border border-slate-200'} backdrop-blur rounded-lg shadow-lg text-xs`}
            style={{ minWidth: '180px', maxWidth: '220px' }}
          >
            {/* Header - always visible */}
            <button
              onClick={() => setLegendExpanded(!legendExpanded)}
              className={`w-full flex items-center justify-between p-3 ${
                darkMode ? 'text-white hover:bg-zinc-800' : 'text-slate-800 hover:bg-slate-100'
              } rounded-lg transition-colors`}
            >
              <span className="font-semibold">Témata ({collectionInfo.topics.length})</span>
              {legendExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            
            {/* Expandable content */}
            {legendExpanded && (
              <div className="px-3 pb-3 max-h-[50vh] overflow-y-auto">
                {/* Show all button */}
                <button
                  onClick={() => setSelectedTopic(null)}
                  className={`w-full mb-2 px-2 py-1.5 rounded text-xs font-medium transition-colors ${
                    selectedTopic === null
                      ? (darkMode ? 'bg-blue-600 text-white' : 'bg-blue-500 text-white')
                      : (darkMode ? 'bg-zinc-700 hover:bg-zinc-600 text-zinc-300' : 'bg-slate-200 hover:bg-slate-300 text-slate-600')
                  }`}
                >
                  Zobrazit vše
                </button>
                
                <div className="space-y-0.5">
                  {collectionInfo.topics.map((topic) => {
                    const isSelected = selectedTopic === topic.id
                    return (
                      <button
                        key={topic.id}
                        onClick={() => setSelectedTopic(isSelected ? null : topic.id)}
                        className={`w-full flex items-center gap-2 px-1.5 py-1 rounded transition-colors ${
                          isSelected 
                            ? (darkMode ? 'bg-zinc-700' : 'bg-blue-100') 
                            : (darkMode ? 'hover:bg-zinc-800' : 'hover:bg-slate-100')
                        }`}
                      >
                        <div
                          className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                          style={{ 
                            backgroundColor: CLUSTER_HEX[topic.id === -1 ? 8 : Math.abs(topic.id) % CLUSTER_HEX.length] 
                          }}
                        />
                        <div className={`flex-1 truncate text-left ${darkMode ? 'text-white' : 'text-slate-700'}`}>
                          {topic.name}
                        </div>
                        <div className={`text-[10px] ${darkMode ? 'text-zinc-500' : 'text-slate-400'}`}>
                          {topic.count}
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )}
        
        {/* Theme toggle - top right */}
        <Button
          onClick={() => setDarkMode(!darkMode)}
          variant="ghost"
          size="icon"
          className={`absolute top-4 right-4 z-30 ${darkMode ? 'bg-zinc-900/90 hover:bg-zinc-800 text-white' : 'bg-white/95 hover:bg-slate-100 text-slate-700 border border-slate-200'} backdrop-blur shadow-sm`}
        >
          {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </Button>
        
        {/* Tooltip */}
        {hoveredPoint && tooltipPos && (
          <div
            className={`absolute pointer-events-none z-50 max-w-sm backdrop-blur border-2 rounded-lg shadow-xl p-3 ${
              darkMode ? 'bg-zinc-900/95 text-white' : 'bg-white/98 text-slate-900 shadow-2xl'
            }`}
            style={{
              left: Math.min(tooltipPos.x + 15, windowSize.width - 380),
              top: Math.min(tooltipPos.y + 15, windowSize.height - 280),
              borderColor: CLUSTER_HEX[Math.abs(hoveredPoint.cluster ?? 0) % CLUSTER_HEX.length],
            }}
          >
            {/* Header with topic and category */}
            <div className="flex items-center gap-2 mb-2 flex-wrap">
              <span 
                className="px-2 py-0.5 rounded text-xs font-bold text-white"
                style={{ 
                  backgroundColor: CLUSTER_HEX[Math.abs(hoveredPoint.cluster ?? 0) % CLUSTER_HEX.length] 
                }}
              >
                {hoveredPoint.topic_name || `Cluster ${hoveredPoint.cluster}`}
              </span>
              {hoveredPoint.metadata?.category && (
                <span className={`text-xs px-2 py-0.5 rounded ${darkMode ? 'bg-zinc-700 text-zinc-300' : 'bg-slate-200 text-slate-600'}`}>
                  {String(hoveredPoint.metadata.category)}
                </span>
              )}
            </div>
            
            {/* Title */}
            {hoveredPoint.metadata?.title && (
              <div className={`font-semibold text-sm mb-2 line-clamp-2 ${darkMode ? 'text-white' : 'text-slate-800'}`}>
                {String(hoveredPoint.metadata.title)}
              </div>
            )}
            
            {/* Text content */}
            <div className={`text-xs line-clamp-3 ${darkMode ? 'text-zinc-400' : 'text-slate-500'}`}>
              {hoveredPoint.text?.replace(/^[.\s]+/, '') || "(žádný text)"}
            </div>
            
            {/* Footer with ID and position */}
            <div className={`text-[10px] font-mono border-t pt-2 mt-2 flex justify-between ${darkMode ? 'text-zinc-500 border-zinc-700' : 'text-slate-400 border-slate-200'}`}>
              <span>ID: {hoveredPoint.id.slice(0, 8)}...</span>
              <span>[{hoveredPoint.x.toFixed(2)}, {hoveredPoint.y.toFixed(2)}, {hoveredPoint.z.toFixed(2)}]</span>
            </div>
          </div>
        )}
      </div>
    </main>
  )
}
