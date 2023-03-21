import './App.css';
import { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sky } from '@react-three/drei';
import CustomMesh from './components/CustomMesh';
import Ocean from './components/Ocean';

function App() {
  return (
    <div id="canvas-container" style={{ width: "100vw", height: "100vh" }}>
      <Canvas camera={{ position: [0, 15, 50], fov: 55, near: 1, far: 20000 }}>
        <Suspense fallback={null}>
          <ambientLight intensity={0.1} />
          <directionalLight color="red" position={[0, 0, 5]} />
          <OrbitControls makeDefault />
          <Sky />
          <Ocean />
          <CustomMesh />
        </Suspense>
      </Canvas>
    </div>
  );
}

export default App;
