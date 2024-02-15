using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.Linq;
using System;
using UnityEngine.Windows.WebCam;
using System.Threading.Tasks;

public class ModelCode : MonoBehaviour
{
    public Texture2D texture;
    public NNModel modelAsset;
    private Model _runtimeModel;
    private IWorker _engine;
    private WebCamTexture webCamTexture;
    private Vector2Int actualCameraSize;

    [Serializable]
    public struct Prediction
    {
        public int predictedValue;
        public float[] predicted;

        public void SetPrediction(Tensor t)
        {
            predicted = t.AsFloats();
            predictedValue = Array.IndexOf(predicted, predicted.Max());
            Debug.Log($"Predicted {predictedValue}");
        }
    }

    public Prediction prediction;


    // Start is called before the first frame update
    void Start()
    {
        webCamTexture = new WebCamTexture("camera", 896, 504, 4);
        webCamTexture.Play();
        _runtimeModel = ModelLoader.Load(modelAsset);
        _engine = WorkerFactory.CreateWorker(_runtimeModel, WorkerFactory.Device.GPU);
        prediction = new Prediction();

    }

    private async Task StartRecognizingAsync()
    {
        await Task.Delay(1000);

        actualCameraSize = new Vector2Int(webCamTexture.width, webCamTexture.height);
        var renderTexture = new RenderTexture(48, 48, 24);

        while (true)
        {
            var cameraTransform = Camera.main.CopyCameraTransForm();
            Graphics.Blit(webCamTexture, renderTexture);
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            // making a tensor out of a grayscale texture
            var channelCount = 1; //grayscale, 3 = color, 4 = color+alpha
            // Create a tensor for input from the texture.
            var inputX = new Tensor(texture, channelCount);

            // Peek at the output tensor without copying it.
            Tensor outputY = _engine.Execute(inputX).PeekOutput();
            // Set the values of our prediction struct using our output tensor.
            prediction.SetPrediction(outputY);

            // Dispose of the input tensor manually (not garbage-collected).
            inputX.Dispose();

        }
    }

    private void OnDestroy()
    {
        // Dispose of the engine manually (not garbage-collected).
        _engine?.Dispose();
    }

}
