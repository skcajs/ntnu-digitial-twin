import { useLoader } from '@react-three/fiber';
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";

function Boat() {
    const gltf = useLoader(GLTFLoader, "./origami_ships.glb");
    return (
        <>
            <primitive position={[0, 0.01, 0]} object={gltf.scene} scale={10} />
        </>
    );
};

export default Boat;


