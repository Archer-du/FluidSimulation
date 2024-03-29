Shader "Spheres"
{
    Properties
    {
        _PrimaryColor ("Primary Color", Color) = (1,1,1,1)
        _SecondaryColor ("Secondary Color", Color) = (1,1,1,1)
        _FoamColor ("Foam Color", Color) = (1,1,1,1)
        [HDR] _SpecularColor ("Specular Color", Color) = (1,1,1,1)
        _PhongExponent ("Phong Exponent", Float) = 128
        _EnvMap ("Environment Map", Cube) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }

        // pass 0: sphere粒子渲染-仅执行顶点从世界坐标到裁剪坐标的变换
        Pass
        {
            CGPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct Particle {
                float4 pos;
                float4 vel;
            };

            StructuredBuffer<Particle> particles;

            float radius;

            struct appdata
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
            };
            
            v2f vert (appdata v, uint id : SV_InstanceID)
            {
                float3 spherePos = particles[id].pos.xyz;
                float3 localPos = v.vertex.xyz * (radius * 2 * 2);

                float3 worldPos = localPos + spherePos;

                v2f o;
                o.vertex = mul(UNITY_MATRIX_VP, float4(worldPos, 1));
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                return 0;
            }
            ENDCG
        }

        // pass 1: 输出世界坐标缓冲
        Pass
        {
            ZTest Always

            CGPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            sampler2D depthBuffer;
            float4x4 inverseV, inverseP;

            float radius;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = v.vertex;
                o.uv = v.uv;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                float d = tex2D(depthBuffer, i.uv);

                float3 viewSpaceRayDir = normalize(mul(inverseP, float4(i.uv*2-1, 0, 1)).xyz);
                float viewSpaceDistance = LinearEyeDepth(d) / dot(viewSpaceRayDir, float3(0,0,-1));

                float3 viewSpacePos = viewSpaceRayDir * viewSpaceDistance;
                float3 worldSpacePos = mul(inverseV, float4(viewSpacePos, 1)).xyz;

                return float4(worldSpacePos, 0);
            }

            ENDCG
        }

        // pass 2: 输出法向缓冲和密度速度纹理
        Pass
        {
            ZTest Less
            ZWrite Off
            Blend One One

            CGPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct Particle {
                float4 pos;
                float4 vel;
            };

            StructuredBuffer<Particle> particles;

            float radius;

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
            };

            float invlerp(float a, float b, float t) {
                return (t-a)/(b-a);
            }
            
            struct v2f
            {
                float4 vertex : SV_POSITION;
                float3 normal : NORMAL;
                float2 densitySpeed : TEXCOORD3;
            };

            struct output2
            {
                float4 normal : SV_Target0;
                float2 densitySpeed : SV_Target1;
            };
            
            v2f vert (appdata v, uint id : SV_InstanceID)
            {
                float3 spherePos = particles[id].pos.xyz;
                float3 localPos = v.vertex.xyz * (radius * 2 * 2);

                float3 worldPos = localPos + spherePos;

                v2f o;
                o.vertex = mul(UNITY_MATRIX_VP, float4(worldPos, 1));
                o.normal = UnityObjectToWorldNormal(v.normal);
                
                // TODO: config
                float upperBound = particles[id].vel.y > 0 ? 100 : 600;
                o.densitySpeed = saturate(float2(invlerp(0, 2, particles[id].pos.w), invlerp(10, upperBound, length(particles[id].vel.xyz))));

                return o;
            }

            output2 frag (v2f i) : SV_Target
            {
                output2 o;
                o.normal = float4(i.normal, 1);
                o.densitySpeed = float2(i.densitySpeed);
                return o;
            }
            ENDCG
        }

        // pass 3
        Pass
        {
            CGPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            sampler2D depthBuffer;
            sampler2D worldPosBuffer;
            sampler2D normalBuffer;
            sampler2D colorBuffer;
            samplerCUBE _EnvMap;

            float4 _PrimaryColor, _SecondaryColor, _FoamColor;
            float4 _SpecularColor;
            float _PhongExponent;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = v.vertex;
                o.vertex.z = 0.5;
                o.uv = v.uv;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                float depth = tex2D(depthBuffer, i.uv);
                float3 worldPos = tex2D(worldPosBuffer, i.uv).xyz;
                float3 normal = tex2D(normalBuffer, i.uv);
                float2 densitySpeed = tex2D(colorBuffer, i.uv);

                if (depth == 0) discard;

                //TODO
                normal = normalize(normal);
                densitySpeed = normalize(densitySpeed);

                float3 lightDir = _WorldSpaceLightPos0.xyz;
                
                // 根据物理信息计算水体漫反射颜色
                float3 ambient = lerp(_PrimaryColor, _FoamColor, densitySpeed.y);
                float3 diffuse = lerp(_PrimaryColor, _SecondaryColor, max(dot(normal, lightDir), 0));

                float3 viewDir = normalize(_WorldSpaceCameraPos.xyz - worldPos);
                float3 reflectDir = reflect(-lightDir, normal);

                float3 result = ambient + diffuse;
                // 镜面高光
                float3 specular = pow(max(dot(viewDir, reflectDir), 0), _PhongExponent) * _SpecularColor;
                result += specular;
                
                float4 reflectedColor = texCUBE(_EnvMap, reflect(-viewDir, normal));

                // Schlick's approximation
                float iorAir = 1.0;
                float iorWater = 1.333;
                float r0 = pow((iorAir - iorWater) / (iorAir + iorWater), 2);
                float rTheta = r0 + (1 - r0) * pow(1 - max(dot(viewDir, normal), 0), 5);

                result = lerp(result, reflectedColor, rTheta);

                return float4(result, 1);
            }
            ENDCG
        }
    }
}
