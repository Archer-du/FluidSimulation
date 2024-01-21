using UnityEngine;

public class CameraController : MonoBehaviour
{
    public float speed = 20;
    public float sensitivity = 7;
    private Camera Cam => Camera.main;
    void Update()
    {
        Vector3 moveDelta = Cam.transform.forward * Input.GetAxis("Vertical") + Cam.transform.right * Input.GetAxis("Horizontal");

        Cam.transform.position += moveDelta * (speed * Time.deltaTime);

        if (Input.GetMouseButton(1)) {
            Cam.transform.Rotate(new Vector3(-Input.GetAxis("Mouse Y"), 0, 0) * sensitivity, Space.Self);
            Cam.transform.Rotate(new Vector3(0, Input.GetAxis("Mouse X"), 0) * sensitivity, Space.World);
        }
    }
}
