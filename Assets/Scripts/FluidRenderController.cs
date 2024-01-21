using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Serialization;

public class FluidRenderController : MonoBehaviour
{
    private struct Particle 
    {
        public Vector4 pos;
        public Vector4 vel;
    }
    
    private const int NumHashes = 1<<20;
    
    /// <summary>
    /// 线程组大小
    /// </summary>
    private const int NumThreads = 1<<10;
    
    /// <summary>
    /// 粒子数量
    /// </summary>
    public int numParticles = 1<<20;
    
    public float deltaTime = 0.001f;

    /// <summary>
    /// 水体初始大小 //TODO: 改成球体
    /// </summary>
    public float initSize = 10;
    
    
    public float gravity = 9.8f;
    public float radius = 1;
    public float gasConstant = 2000;
    public float restDensity = 10;
    public float mass = 1;
    public float density = 1;
    public float viscosity = 0.01f;

    // 物理边界
    public Vector3 minBounds = new Vector3(-10, -10, -10);
    public Vector3 maxBounds = new Vector3(10, 10, 10);

    public ComputeShader fluidComputer;

    [FormerlySerializedAs("renderMat")] public Material fluidMaterial;

    /// <summary>
    /// 渲染指令缓冲
    /// </summary>
    private CommandBuffer commandBuffer;
    
    private Mesh screenQuadMesh;
    public Mesh sphereMesh;
    public Mesh particleMesh;
    
    public float particleRenderSize = 0.4f;

    public Color primaryColor;
    public Color secondaryColor;
    
    public int moveParticles = 10;

    private ComputeBuffer hashesBuffer;
    private ComputeBuffer globalHashCounterBuffer;
    private ComputeBuffer localIndicesBuffer;
    private ComputeBuffer inverseIndicesBuffer;
    private ComputeBuffer particlesBuffer;
    private ComputeBuffer sortedBuffer;
    private ComputeBuffer forcesBuffer;
    private ComputeBuffer groupArrBuffer;
    private ComputeBuffer hashDebugBuffer;
    private ComputeBuffer hashValueDebugBuffer;
    private ComputeBuffer meanBuffer;
    private ComputeBuffer covBuffer;
    private ComputeBuffer hashRangeBuffer;
    
    // 渲染参数
    private ComputeBuffer quadInstancedArgsBuffer;
    private ComputeBuffer sphereInstancedArgsBuffer;

    private int solverFrame = 0;

    private int moveParticleBeginIndex = 0;

    private double lastFrameTimestamp = 0;
    private double totalFrameTime = 0;

    private Vector4[] CurrPlanes => blocked ? boxPlanes : groundPlanes;
    private Vector4[] boxPlanes = new Vector4[7];
    private Vector4[] groundPlanes = new Vector4[7];


    // 输入控制参数
    private bool paused = false;
    private bool blocked = true;

    
    private static Vector4 GetPlaneParam(Vector3 p, Vector3 n) => new(n.x, n.y, n.z, -Vector3.Dot(p, n));

    void Start() {
        Particle[] particles = new Particle[numParticles];

        Vector3 origin1 = new Vector3(
            Mathf.Lerp(minBounds.x, maxBounds.x, 0.3f),
            Mathf.Lerp(minBounds.y, maxBounds.y, 0.6f),
            Mathf.Lerp(minBounds.z, maxBounds.z, 0.3f)
        );
        Vector3 origin2 = new Vector3(
            Mathf.Lerp(minBounds.x, maxBounds.x, 0.7f),
            Mathf.Lerp(minBounds.y, maxBounds.y, 0.6f),
            Mathf.Lerp(minBounds.z, maxBounds.z, 0.7f)
        );

        for (int i = 0; i < numParticles; i++) {
            Vector3 pos = new Vector3(
                (Random.Range(0f, 1f) - 0.5f) * initSize,
                (Random.Range(0f, 1f) - 0.5f) * initSize,
                (Random.Range(0f, 1f) - 0.5f) * initSize
            );

            pos += i % 2 == 0 ? origin1 : origin2;

            particles[i].pos = pos;
        }
        
        float poly6 = 315f / (64f * Mathf.PI * Mathf.Pow(radius, 9f));
        float spiky = 45f / (Mathf.PI * Mathf.Pow(radius, 6f));
        float visco = 45f / (Mathf.PI * Mathf.Pow(radius, 6f));
        
        fluidComputer.SetInt("numHash", NumHashes);
        fluidComputer.SetInt("numParticles", numParticles);

        fluidComputer.SetFloat("radiusSqr", radius * radius);
        fluidComputer.SetFloat("radius", radius);
        fluidComputer.SetFloat("gasConst", gasConstant);
        fluidComputer.SetFloat("restDensity", restDensity);
        fluidComputer.SetFloat("mass", mass);
        fluidComputer.SetFloat("viscosity", viscosity);
        fluidComputer.SetFloat("gravity", gravity);
        fluidComputer.SetFloat("deltaTime", deltaTime);

        fluidComputer.SetFloat("poly6Coeff", poly6);
        fluidComputer.SetFloat("spikyCoeff", spiky);
        fluidComputer.SetFloat("viscoCoeff", visco * viscosity);

        boxPlanes[0] = GetPlaneParam(new Vector3(0, minBounds.y, 0), Vector3.up);
        boxPlanes[1] = GetPlaneParam(new Vector3(0, maxBounds.y, 0), Vector3.down);
        boxPlanes[2] = GetPlaneParam(new Vector3(minBounds.x, 0, 0), Vector3.right);
        boxPlanes[3] = GetPlaneParam(new Vector3(maxBounds.x, 0, 0), Vector3.left);
        boxPlanes[4] = GetPlaneParam(new Vector3(0, 0, minBounds.z), Vector3.forward);
        boxPlanes[5] = GetPlaneParam(new Vector3(0, 0, maxBounds.z), Vector3.back);
        groundPlanes[0] = GetPlaneParam(new Vector3(0, 0, 0), Vector3.up);
        groundPlanes[1] = GetPlaneParam(new Vector3(0, 100, 0), Vector3.down);
        
        hashesBuffer = new ComputeBuffer(numParticles, 4);

        globalHashCounterBuffer = new ComputeBuffer(NumHashes, 4);

        localIndicesBuffer = new ComputeBuffer(numParticles, 4);

        inverseIndicesBuffer = new ComputeBuffer(numParticles, 4);

        particlesBuffer = new ComputeBuffer(numParticles, 4 * 8);
        particlesBuffer.SetData(particles);

        sortedBuffer = new ComputeBuffer(numParticles, 4 * 8);

        forcesBuffer = new ComputeBuffer(numParticles * 2, 4 * 4);

        int groupArrLen = Mathf.CeilToInt(NumHashes / 1024f);
        groupArrBuffer = new ComputeBuffer(groupArrLen, 4);

        hashDebugBuffer = new ComputeBuffer(4, 4);
        hashValueDebugBuffer = new ComputeBuffer(numParticles, 4 * 3);

        meanBuffer = new ComputeBuffer(numParticles, 4 * 4);
        covBuffer = new ComputeBuffer(numParticles, 4 * 3 * 2);
        hashRangeBuffer = new ComputeBuffer(NumHashes, 4 * 2);

        for (int i = 0; i < 12; i++) {
            fluidComputer.SetBuffer(i, "hashes", hashesBuffer);
            fluidComputer.SetBuffer(i, "globalHashCounter", globalHashCounterBuffer);
            fluidComputer.SetBuffer(i, "localIndices", localIndicesBuffer);
            fluidComputer.SetBuffer(i, "inverseIndices", inverseIndicesBuffer);
            fluidComputer.SetBuffer(i, "particles", particlesBuffer);
            fluidComputer.SetBuffer(i, "sorted", sortedBuffer);
            fluidComputer.SetBuffer(i, "forces", forcesBuffer);
            fluidComputer.SetBuffer(i, "groupArr", groupArrBuffer);
            fluidComputer.SetBuffer(i, "hashDebug", hashDebugBuffer);
            fluidComputer.SetBuffer(i, "mean", meanBuffer);
            fluidComputer.SetBuffer(i, "cov", covBuffer);
            fluidComputer.SetBuffer(i, "hashRange", hashRangeBuffer);
            fluidComputer.SetBuffer(i, "hashValueDebug", hashValueDebugBuffer);
        }

        fluidMaterial.SetBuffer("particles", particlesBuffer);
        fluidMaterial.SetFloat("radius", particleRenderSize * 0.5f);

        quadInstancedArgsBuffer = new ComputeBuffer(1, sizeof(uint) * 5, ComputeBufferType.IndirectArguments);

        uint[] args = new uint[5];
        args[0] = particleMesh.GetIndexCount(0);
        args[1] = (uint) numParticles;
        args[2] = particleMesh.GetIndexStart(0);
        args[3] = particleMesh.GetBaseVertex(0);
        args[4] = 0;

        quadInstancedArgsBuffer.SetData(args);

        sphereInstancedArgsBuffer = new ComputeBuffer(1, sizeof(uint) * 5, ComputeBufferType.IndirectArguments);

        uint[] args2 = new uint[5];
        args2[0] = sphereMesh.GetIndexCount(0);
        args2[1] = (uint) numParticles;
        args2[2] = sphereMesh.GetIndexStart(0);
        args2[3] = sphereMesh.GetBaseVertex(0);
        args2[4] = 0;

        sphereInstancedArgsBuffer.SetData(args2);

        screenQuadMesh = new Mesh
        {
            vertices = new Vector3[] {
                new( 1.0f , 1.0f,  0.0f),
                new(-1.0f , 1.0f,  0.0f),
                new(-1.0f ,-1.0f,  0.0f),
                new( 1.0f ,-1.0f,  0.0f),
            },
            uv = new Vector2[] {
                new(1, 0),
                new(0, 0),
                new(0, 1),
                new(1, 1)
            },
            triangles = new int[6] { 0, 1, 2, 2, 3, 0 }
        };

        commandBuffer = new CommandBuffer();
        commandBuffer.name = "Fluid Render";

        InitCommandBuffer();
        Camera.main.AddCommandBuffer(CameraEvent.AfterForwardAlpha, commandBuffer);
    }

    void Update() {
        {
            ProcessInput();
            fluidComputer.SetVectorArray("planes", CurrPlanes);

            if (Input.GetMouseButton(0)) {
                Ray mouseRay = Camera.main.ScreenPointToRay(Input.mousePosition);
                if (Physics.Raycast(mouseRay, out RaycastHit hit)) {
                    Vector3 pos = new Vector3(
                        Mathf.Clamp(hit.point.x, minBounds.x, maxBounds.x),
                        maxBounds.y - 1f,
                        Mathf.Clamp(hit.point.z, minBounds.z, maxBounds.z)
                    );

                    fluidComputer.SetInt("moveBeginIndex", moveParticleBeginIndex);
                    fluidComputer.SetInt("moveSize", moveParticles);
                    fluidComputer.SetVector("movePos", pos);
                    fluidComputer.SetVector("moveVel", Vector3.down * 70);

                    fluidComputer.Dispatch(fluidComputer.FindKernel("MoveParticles"), 1, 1, 1);

                    moveParticleBeginIndex = (moveParticleBeginIndex + moveParticles * moveParticles) % numParticles;
                }
            }

            double solverStart = Time.realtimeSinceStartupAsDouble;

            fluidComputer.Dispatch(fluidComputer.FindKernel("ResetCounter"), Mathf.CeilToInt((float)NumHashes / NumThreads), 1, 1);
            fluidComputer.Dispatch(fluidComputer.FindKernel("InsertToBucket"), Mathf.CeilToInt((float)numParticles / NumThreads), 1, 1);

            
            
            fluidComputer.Dispatch(fluidComputer.FindKernel("PrefixSum1"), Mathf.CeilToInt((float)NumHashes / NumThreads), 1, 1);

            // @Important: Because of the way prefix sum algorithm implemented,
            // Currently maximum numHashes value is numThreads^2.
            Debug.Assert(NumHashes <= NumThreads*NumThreads);
            fluidComputer.Dispatch(fluidComputer.FindKernel("PrefixSum2"), 1, 1, 1);

            fluidComputer.Dispatch(fluidComputer.FindKernel("PrefixSum3"), Mathf.CeilToInt((float)NumHashes / NumThreads), 1, 1);
            fluidComputer.Dispatch(fluidComputer.FindKernel("Sort"), Mathf.CeilToInt((float)numParticles / NumThreads), 1, 1);
            fluidComputer.Dispatch(fluidComputer.FindKernel("CalcHashRange"), Mathf.CeilToInt((float)NumHashes / NumThreads), 1, 1);

            if (!paused) {
                for (int iter = 0; iter < 1; iter++) {
                    fluidComputer.Dispatch(fluidComputer.FindKernel("CalcPressure"), Mathf.CeilToInt((float)numParticles / 128), 1, 1);
                    fluidComputer.Dispatch(fluidComputer.FindKernel("CalcForces"), Mathf.CeilToInt((float)numParticles / 128), 1, 1);
                    fluidComputer.Dispatch(fluidComputer.FindKernel("Step"), Mathf.CeilToInt((float)numParticles / NumThreads), 1, 1);
                }

                solverFrame++;

                if (solverFrame > 1) {
                    totalFrameTime += Time.realtimeSinceStartupAsDouble - lastFrameTimestamp;
                }

                if (solverFrame == 400 || solverFrame == 1200) {
                    Debug.Log($"Avg frame time at #{solverFrame}: {totalFrameTime / (solverFrame-1) * 1000}ms.");
                }
            }

            lastFrameTimestamp = Time.realtimeSinceStartupAsDouble;
        }
    }

    void InitCommandBuffer() {
        commandBuffer.Clear();

        int[] worldPosBufferIds = {
            Shader.PropertyToID("worldPosBuffer1"),
            Shader.PropertyToID("worldPosBuffer2")
        };

        // pass 0
        commandBuffer.GetTemporaryRT(worldPosBufferIds[0], Screen.width, Screen.height, 0, FilterMode.Point, RenderTextureFormat.ARGBFloat);

        int depthId = Shader.PropertyToID("depthBuffer");
        commandBuffer.GetTemporaryRT(depthId, Screen.width, Screen.height, 32, FilterMode.Point, RenderTextureFormat.Depth);

        commandBuffer.SetRenderTarget((RenderTargetIdentifier)worldPosBufferIds[0], (RenderTargetIdentifier)depthId);
        commandBuffer.ClearRenderTarget(true, true, Color.clear);

        commandBuffer.DrawMeshInstancedIndirect(
            sphereMesh,
            0,
            fluidMaterial,
            0,
            sphereInstancedArgsBuffer
        );

        //pass 1
        int depth2Id = Shader.PropertyToID("depth2Buffer");
        commandBuffer.GetTemporaryRT(depth2Id, Screen.width, Screen.height, 32, FilterMode.Point, RenderTextureFormat.Depth);

        commandBuffer.SetRenderTarget((RenderTargetIdentifier)worldPosBufferIds[0], (RenderTargetIdentifier)depth2Id);
        commandBuffer.ClearRenderTarget(true, true, Color.clear);

        commandBuffer.SetGlobalTexture("depthBuffer", depthId);

        commandBuffer.DrawMesh(
            screenQuadMesh,
            Matrix4x4.identity,
            fluidMaterial,
            0,
            1
        );
        
        //pass 2
        int normalBufferId = Shader.PropertyToID("normalBuffer");
        commandBuffer.GetTemporaryRT(normalBufferId, Screen.width, Screen.height, 0, FilterMode.Point, RenderTextureFormat.ARGBHalf);

        int colorBufferId = Shader.PropertyToID("colorBuffer");
        commandBuffer.GetTemporaryRT(colorBufferId, Screen.width, Screen.height, 0, FilterMode.Point, RenderTextureFormat.RGHalf);

        commandBuffer.SetRenderTarget(new RenderTargetIdentifier[] { normalBufferId, colorBufferId }, (RenderTargetIdentifier)depthId);
        commandBuffer.ClearRenderTarget(false, true, Color.clear);

        commandBuffer.SetGlobalTexture("worldPosBuffer", worldPosBufferIds[0]);

        commandBuffer.DrawMeshInstancedIndirect(
            particleMesh,
            0,
            fluidMaterial,
            2,
            quadInstancedArgsBuffer
        );
        
        //pass 3
        commandBuffer.SetGlobalTexture("normalBuffer", normalBufferId);
        commandBuffer.SetGlobalTexture("colorBuffer", colorBufferId);

        commandBuffer.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);

        commandBuffer.DrawMesh(
            screenQuadMesh,
            Matrix4x4.identity,
            fluidMaterial,
            0,
            3
        );
    }
    
    void LateUpdate() {
        Matrix4x4 view = Camera.main.worldToCameraMatrix;

        Shader.SetGlobalMatrix("inverseV", view.inverse);
        Shader.SetGlobalMatrix("inverseP", Camera.main.projectionMatrix.inverse);
    }
    
    private void ProcessInput()
    {
        if (Input.GetKeyDown(KeyCode.X)) {
            blocked = !blocked;
        }
        if (Input.GetKeyDown(KeyCode.Space)) {
            paused = !paused;
        }
        if (Input.GetKeyDown(KeyCode.C)) {
            uint[] debugResult = new uint[4];

            hashDebugBuffer.SetData(debugResult);

            fluidComputer.Dispatch(fluidComputer.FindKernel("DebugHash"), Mathf.CeilToInt((float)NumHashes / NumThreads), 1, 1);

            hashDebugBuffer.GetData(debugResult);

            uint usedHashBuckets = debugResult[0];
            uint maxSameHash = debugResult[1];

            Debug.Log($"Total buckets: {NumHashes}, Used buckets: {usedHashBuckets}, Used rate: {(float)usedHashBuckets / NumHashes * 100}%");
            Debug.Log($"Avg hash collision: {(float)numParticles / usedHashBuckets}, Max hash collision: {maxSameHash}");
        }
        // Debug
        if (Input.GetKeyDown(KeyCode.C)) {
            uint[] debugResult = new uint[4];

            int[] values = new int[numParticles * 3];

            hashDebugBuffer.SetData(debugResult);

            fluidComputer.Dispatch(fluidComputer.FindKernel("DebugHash"), Mathf.CeilToInt((float)NumHashes / NumThreads), 1, 1);

            hashDebugBuffer.GetData(debugResult);

            uint totalAccessCount = debugResult[2];
            uint totalNeighborCount = debugResult[3];

            Debug.Log($"Total access: {totalAccessCount}, Avg access: {(float)totalAccessCount / numParticles}, Avg accept: {(float)totalNeighborCount / numParticles}");
            Debug.Log($"Average accept rate: {(float)totalNeighborCount / totalAccessCount * 100}%");

            hashValueDebugBuffer.GetData(values);

            HashSet<Vector3Int> set = new HashSet<Vector3Int>();
            for (int i = 0; i < numParticles; i++) {
                Vector3Int vi = new Vector3Int(values[i*3+0], values[i*3+1], values[i*3+2]);
                set.Add(vi);
            }

            Debug.Log($"Total unique hash keys: {set.Count}, Ideal bucket load: {(float)set.Count / NumHashes * 100}%");
        }
    }

    void OnDisable() {
        hashesBuffer.Dispose();
        globalHashCounterBuffer.Dispose();
        localIndicesBuffer.Dispose();
        inverseIndicesBuffer.Dispose();
        particlesBuffer.Dispose();
        sortedBuffer.Dispose();
        forcesBuffer.Dispose();
        groupArrBuffer.Dispose();
        hashDebugBuffer.Dispose();
        hashValueDebugBuffer.Dispose();
        meanBuffer.Dispose();
        covBuffer.Dispose();
        hashRangeBuffer.Dispose();

        quadInstancedArgsBuffer.Dispose();
        sphereInstancedArgsBuffer.Dispose();
    }
}
