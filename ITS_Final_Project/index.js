async function runExample() {
    var x = [
        parseFloat(document.getElementById('box1').value),
        parseFloat(document.getElementById('box2').value),
        parseFloat(document.getElementById('box3').value),
        parseFloat(document.getElementById('box4').value),
        parseFloat(document.getElementById('box5').value),
        parseFloat(document.getElementById('box6').value),
        parseFloat(document.getElementById('box7').value)
    ];

    let tensorX = new onnx.Tensor(x, 'float32', [1, 7]);

    let session = new onnx.InferenceSession();

    await session.loadModel("./DLnet_Diabetes.onnx");
    let outputMap = await session.run([tensorX]);
    let outputData = outputMap.get('output1');

    let predictions = document.getElementById('predictions');

    predictions.innerHTML = `<hr> Got an output tensor with values: <br/>
    <table>
        <tr>
            <td>Diabetes Predictor</td>
            <td id="td0">${outputData.data[0].toFixed(2)}</td>
        </tr>
    </table>`;
}
