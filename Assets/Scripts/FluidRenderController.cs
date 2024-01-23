using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class FluidRenderController : MonoBehaviour
{
    private struct Particle 
    {
        public Vector4 pos;
        public Vector4 vel;
    }
    
    // global ------------------------------------------------------
    private Camera Cam => Camera.main;
    
    /// <summary>
    /// compute shader kernel函数数量
    /// </summary>
    private const int KernelNum = 11;
    
    /// <summary>
    /// 线程组大小
    /// </summary>
    private const int NumThreads = 1<<10;
    
    private const int NumHashes = 1<<20;
    
    /// <summary>
    /// 粒子数量
    /// </summary>
    public int numParticles = 1<<20;
    
    public float deltaTime = 0.001f;

    
    // physics -----------------------------------------------------
    /// <summary>
    /// 水体初始大小 //TODO: 改成球体
    /// </summary>
    public float initSize = 10;
    
    // 物理解算参数
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
    private Vector4[] CurrPlanes => blocked ? boxPlanes : groundPlanes;
    private Vector4[] boxPlanes = new Vector4[7];
    private Vector4[] groundPlanes = new Vector4[7];

    public Vector3[] originNormalizedDropPoints;

    /// <summary>
    /// compute shader
    /// </summary>
    public ComputeShader fluidComputer;

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
    private ComputeBuffer hashRangeBuffer;
    
    
    // render ------------------------------------------------------
    /// <summary>
    /// 渲染指令缓冲
    /// </summary>
    private CommandBuffer commandBuffer;
    
    private Mesh screenQuadMesh;
    public Mesh sphereMesh;
    public Mesh particleMesh;
    
    public float particleRenderSize = 0.4f;

    /// <summary>
    /// vertex & fragment shader
    /// </summary>
    public Material fluidMaterial;

    // 渲染参数缓冲
    private ComputeBuffer quadInstancedArgsBuffer;
    private ComputeBuffer sphereInstancedArgsBuffer;

    
    // debug & performance ----------------------------------------
    private int solverFrame = 0;

    private double lastFrameTimestamp = 0;
    private double totalFrameTime = 0;
    
    
    // input  -----------------------------------------------------
    // 输入控制参数
    private bool paused = false;
    private bool blocked = true;

    
    /// <summary>
    /// 获取平面方程参数
    /// </summary>
    /// <param name="p">平面上一点</param>
    /// <param name="n">平面法向</param>
    /// <returns>平面方程参数</returns>
    private Vector4 GetPlaneParam(Vector3 p, Vector3 n) => new(n.x, n.y, n.z, -Vector3.Dot(p, n));

    void Start() 
    {
        // 设置粒子初始位置
        Particle[] particles = new Particle[numParticles];
        Vector3[] origins = new Vector3[originNormalizedDropPoints.Length];

        for (int i = 0; i < originNormalizedDropPoints.Length; i++)
        {
            origins[i] = new Vector3(
                Mathf.Lerp(minBounds.x, maxBounds.x, originNormalizedDropPoints[i].x),
                Mathf.Lerp(minBounds.y, maxBounds.y, originNormalizedDropPoints[i].y),
                Mathf.Lerp(minBounds.z, maxBounds.z, originNormalizedDropPoints[i].z)
            );
        }
        for (int i = 0; i < numParticles; i++) {
            Vector3 pos = new Vector3(
                (Random.Range(0f, 1f) - 0.5f) * initSize,
                (Random.Range(0f, 1f) - 0.5f) * initSize,
                (Random.Range(0f, 1f) - 0.5f) * initSize
            );

            pos += origins[i % originNormalizedDropPoints.Length];
            particles[i].pos = pos;
        }
        hashesBuffer            = new ComputeBuffer(numParticles, 4);
        globalHashCounterBuffer = new ComputeBuffer(NumHashes, 4);
        localIndicesBuffer      = new ComputeBuffer(numParticles, 4);
        inverseIndicesBuffer    = new ComputeBuffer(numParticles, 4);
        particlesBuffer         = new ComputeBuffer(numParticles, 4 * 8);
        particlesBuffer.SetData(particles);
        sortedBuffer            = new ComputeBuffer(numParticles, 4 * 8);
        forcesBuffer            = new ComputeBuffer(numParticles * 2, 4 * 4);
        int groupArrLen = Mathf.CeilToInt(NumHashes / 1024f);
        groupArrBuffer          = new ComputeBuffer(groupArrLen, 4);
        hashDebugBuffer         = new ComputeBuffer(4, 4);
        hashValueDebugBuffer    = new ComputeBuffer(numParticles, 4 * 3);
        hashRangeBuffer         = new ComputeBuffer(NumHashes, 4 * 2);
        SetPhysicsArgs();
        SetRenderArgs();
    }


    void Update() 
    {
        ProcessInput();

        fluidComputer.Dispatch(fluidComputer.FindKernel("ResetCounter"), Mathf.CeilToInt((float)NumHashes / NumThreads), 1, 1);

        fluidComputer.Dispatch(fluidComputer.FindKernel("InsertToBucket"), Mathf.CeilToInt((float)numParticles / NumThreads), 1, 1);
        fluidComputer.Dispatch(fluidComputer.FindKernel("PrefixSum1"), Mathf.CeilToInt((float)NumHashes / NumThreads), 1, 1);
        Debug.Assert(NumHashes <= NumThreads*NumThreads);
        fluidComputer.Dispatch(fluidComputer.FindKernel("PrefixSum2"), 1, 1, 1);
        fluidComputer.Dispatch(fluidComputer.FindKernel("PrefixSum3"), Mathf.CeilToInt((float)NumHashes / NumThreads), 1, 1);
        fluidComputer.Dispatch(fluidComputer.FindKernel("Sort"), Mathf.CeilToInt((float)numParticles / NumThreads), 1, 1);
        
        fluidComputer.Dispatch(fluidComputer.FindKernel("CalcHashRange"), Mathf.CeilToInt((float)NumHashes / NumThreads), 1, 1);

        if (!paused) {
            fluidComputer.Dispatch(fluidComputer.FindKernel("CalcDensity"), Mathf.CeilToInt((float)numParticles / NumThreads), 1, 1);
            fluidComputer.Dispatch(fluidComputer.FindKernel("CalcForces"), Mathf.CeilToInt((float)numParticles / NumThreads), 1, 1);
            fluidComputer.Dispatch(fluidComputer.FindKernel("Step"), Mathf.CeilToInt((float)numParticles / NumThreads), 1, 1);

            solverFrame++;
            if (solverFrame > 1) {
                totalFrameTime += Time.realtimeSinceStartupAsDouble - lastFrameTimestamp;
            }
            if (solverFrame % 500 == 250) {
                Debug.Log($"平均FPS #{solverFrame}: {(solverFrame-1) / totalFrameTime}.");
            }
        }
        lastFrameTimestamp = Time.realtimeSinceStartupAsDouble;
    }
    void LateUpdate() {
        Shader.SetGlobalMatrix("inverseV", Cam.worldToCameraMatrix.inverse);
        Shader.SetGlobalMatrix("inverseP", Cam.projectionMatrix.inverse);
    }
    private void InitCommandBuffer() {
        commandBuffer.Clear();

        int worldPosBufferId = Shader.PropertyToID("worldPosBuffer");
        // pass 0
        commandBuffer.GetTemporaryRT(worldPosBufferId, Screen.width, Screen.height, 0, FilterMode.Point, RenderTextureFormat.ARGBFloat);

        int depthId = Shader.PropertyToID("depthBuffer");
        commandBuffer.GetTemporaryRT(depthId, Screen.width, Screen.height, 32, FilterMode.Point, RenderTextureFormat.Depth);

        commandBuffer.SetRenderTarget((RenderTargetIdentifier)worldPosBufferId, (RenderTargetIdentifier)depthId);
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

        commandBuffer.SetRenderTarget((RenderTargetIdentifier)worldPosBufferId, (RenderTargetIdentifier)depth2Id);
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

        commandBuffer.DrawMeshInstancedIndirect(
            particleMesh,
            0,
            fluidMaterial,
            2,
            quadInstancedArgsBuffer
        );
        
        //pass 3
        commandBuffer.SetGlobalTexture("worldPosBuffer", worldPosBufferId);
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

    
    /// <summary>
    /// 为GPU端compute shader传入物理解算参数
    /// </summary>
    private void SetPhysicsArgs()
    {
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


        for (int kernelId = 0; kernelId < KernelNum; kernelId++) {
            fluidComputer.SetBuffer(kernelId, "hashes", hashesBuffer);
            fluidComputer.SetBuffer(kernelId, "globalHashCounter", globalHashCounterBuffer);
            fluidComputer.SetBuffer(kernelId, "localIndices", localIndicesBuffer);
            fluidComputer.SetBuffer(kernelId, "inverseIndices", inverseIndicesBuffer);
            fluidComputer.SetBuffer(kernelId, "particles", particlesBuffer);
            fluidComputer.SetBuffer(kernelId, "sorted", sortedBuffer);
            fluidComputer.SetBuffer(kernelId, "forces", forcesBuffer);
            fluidComputer.SetBuffer(kernelId, "groupArr", groupArrBuffer);
            fluidComputer.SetBuffer(kernelId, "hashDebug", hashDebugBuffer);
            fluidComputer.SetBuffer(kernelId, "hashRange", hashRangeBuffer);
            fluidComputer.SetBuffer(kernelId, "hashValueDebug", hashValueDebugBuffer);
        }
        
        boxPlanes[0] = GetPlaneParam(new Vector3(0, minBounds.y, 0), Vector3.up);
        boxPlanes[1] = GetPlaneParam(new Vector3(0, maxBounds.y, 0), Vector3.down);
        boxPlanes[2] = GetPlaneParam(new Vector3(minBounds.x, 0, 0), Vector3.right);
        boxPlanes[3] = GetPlaneParam(new Vector3(maxBounds.x, 0, 0), Vector3.left);
        boxPlanes[4] = GetPlaneParam(new Vector3(0, 0, minBounds.z), Vector3.forward);
        boxPlanes[5] = GetPlaneParam(new Vector3(0, 0, maxBounds.z), Vector3.back);
        groundPlanes[0] = GetPlaneParam(new Vector3(0, 0, 0), Vector3.up);
        groundPlanes[1] = GetPlaneParam(new Vector3(0, 100, 0), Vector3.down);
        
        fluidComputer.SetVectorArray("planes", CurrPlanes);
    }

    private void SetRenderArgs()
    {
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
        Cam.AddCommandBuffer(CameraEvent.AfterForwardAlpha, commandBuffer);
    }
    
    
    private void ProcessInput()
    {
        if (Input.GetKeyDown(KeyCode.X)) {
            blocked = false;
            fluidComputer.SetVectorArray("planes", CurrPlanes);
        }
        if (Input.GetKeyDown(KeyCode.Space)) {
            paused = !paused;
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
        hashRangeBuffer.Dispose();

        quadInstancedArgsBuffer.Dispose();
        sphereInstancedArgsBuffer.Dispose();
    }
}
