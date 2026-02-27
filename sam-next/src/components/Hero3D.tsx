'use client'
import { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Sphere, MeshDistortMaterial, Float, Torus } from '@react-three/drei'
import * as THREE from 'three'

function Particle({ position }: { position: [number, number, number] }) {
  const mesh = useRef<THREE.Mesh>(null)
  useFrame((state) => {
    if (!mesh.current) return
    mesh.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 0.8 + position[0]) * 0.3
    mesh.current.rotation.x += 0.005
  })
  return (
    <mesh ref={mesh} position={position}>
      <sphereGeometry args={[0.04, 8, 8]} />
      <meshStandardMaterial color="#0F9D9A" transparent opacity={0.6} />
    </mesh>
  )
}

function Scene() {
  const particles = useMemo(() => {
    return Array.from({ length: 20 }, (_, i) => ({
      position: [
        (Math.random() - 0.5) * 4,
        (Math.random() - 0.5) * 4,
        (Math.random() - 0.5) * 2,
      ] as [number, number, number],
      key: i,
    }))
  }, [])

  return (
    <>
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 5, 5]} intensity={0.8} color="#ffffff" />
      <pointLight position={[-3, 3, 3]} intensity={0.5} color="#0F9D9A" />

      <Float speed={1.4} rotationIntensity={0.6} floatIntensity={0.8}>
        <Sphere args={[1, 64, 64]} position={[0, 0, 0]}>
          <MeshDistortMaterial
            color="#0F9D9A"
            distort={0.35}
            speed={1.5}
            roughness={0.1}
            metalness={0.3}
            transparent
            opacity={0.85}
          />
        </Sphere>
      </Float>

      <Float speed={0.8} rotationIntensity={1.2} floatIntensity={0.4}>
        <Torus args={[1.6, 0.04, 16, 100]} position={[0, 0, 0]} rotation={[Math.PI / 3, 0, 0]}>
          <meshStandardMaterial color="#0F9D9A" transparent opacity={0.3} />
        </Torus>
      </Float>

      <Float speed={1.1} rotationIntensity={0.8} floatIntensity={0.6}>
        <Torus args={[1.9, 0.03, 16, 100]} position={[0, 0, 0]} rotation={[-Math.PI / 4, Math.PI / 6, 0]}>
          <meshStandardMaterial color="#5bbfbd" transparent opacity={0.2} />
        </Torus>
      </Float>

      {particles.map((p) => <Particle key={p.key} position={p.position} />)}
    </>
  )
}

export default function Hero3D() {
  return (
    <div className="w-full h-full" aria-hidden="true">
      <Canvas
        camera={{ position: [0, 0, 4.5], fov: 50 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: 'transparent' }}
      >
        <Scene />
      </Canvas>
    </div>
  )
}
