// import { useLoader } from '@react-three/fiber';
// import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";

// function Boat() {
//     const gltf = useLoader(GLTFLoader, "./origami_boat.glb");
//     return (
//         <>
//             <primitive position={[0, 0.01, 0]} object={gltf.scene} scale={10} />
//         </>
//     );
// };

// export default Boat;



import React, { useRef, useEffect, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import axios from "axios";

export default function Model(props) {
    const [data, setData] = useState({});
    const group = useRef();
    const { nodes, materials } = useGLTF("/origami_boat.glb");

    const myReallyBadHardCodedAPIPath = "http://localhost:4999";

    useEffect(() => {
        axios.get(`${myReallyBadHardCodedAPIPath}/sim`).then(
            function (result) {
                const data = result;
                setData(data.data);
            }
        );
    });

    useFrame(({ clock }) => {
        const a = clock.getElapsedTime();
        const key = String(Math.round(a * 10) / 10);
        if (Object.keys(data).includes(key)) {
            group.current.position.z = data[key]['x'];
            group.current.position.x = data[key]['y'];
            group.current.rotation.y = data[key]['psi'];
        }
    });


    return (
        <group ref={group} {...props} dispose={null} scale={0.4}>
            <mesh
                castShadow
                receiveShadow
                geometry={nodes.Plane015_Ships_0.geometry}
                material={materials.Ships}
                scale={0.5}
            />
        </group>
    );
}

useGLTF.preload("/origami_boat.glb");


// import React, { useRef } from "react";
// import { useGLTF } from "@react-three/drei";

// export default function Model(props) {
//     const { nodes, materials } = useGLTF("/origami_boat.glb");
//     return (
//         <group {...props} dispose={null}>
//             <group rotation={[-Math.PI / 2, 0, 0]} scale={0}>
//                 <group rotation={[Math.PI / 2, 0, 0]}>
//                     <mesh
//                         castShadow
//                         receiveShadow
//                         geometry={nodes.Plane015_Ships_0.geometry}
//                         material={materials.Ships}
//                     />
//                 </group>
//             </group>
//         </group>
//     );
// }

// useGLTF.preload("/origami_boat.glb");