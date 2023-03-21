import './App.css';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sky } from '@react-three/drei';
import CustomMesh from './components/CustomMesh';
import Ocean from './components/Ocean';
import { Suspense } from 'react';

function App() {
  return (
    <div id="canvas-container" style={{ width: "100vw", height: "100vh" }}>
      <Canvas camera={{ position: [0, 5, 50], fov: 55, near: 1, far: 20000 }}>
        <ambientLight intensity={0.1} />
        <directionalLight color="red" position={[0, 0, 5]} />
        <OrbitControls makeDefault />
        <Sky />
        <Suspense fallback={null}>
          <Ocean />
        </Suspense>
        <CustomMesh />
      </Canvas>
    </div>
  );
}

export default App;
