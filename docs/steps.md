### 1. Repository

- **Huggingface** (as Repo)

### 2. System Integration Test

- **SIT Task Manager**
- **Postprocessing Tester**
- **ONNX Converter**
- **Graph Transposer**
- **Holohub**

### 3. Deployment

- (Currently no specific participants or content; can be added as needed)

### 4. Interaction Flow

- **Developer** → **Huggingface**: Push Model with Metadata
- **Huggingface** → **SIT Task Manager**: Raise Event
- **SIT Task Manager** → **Huggingface**: Pull Model and Metadata
- **SIT Task Manager** → **Postprocessing Tester**: Trigger test and send model
- **Postprocessing Tester** → **SIT Task Manager**: Return testing result
- **SIT Task Manager** → **ONNX Converter**: Request ONNX converter with model
- **ONNX Converter** → **ONNX Converter**: Convert .pt model to ONNX
- **ONNX Converter** → **SIT Task Manager**: Response with ONNX model
- **SIT Task Manager** → **Graph Transposer**: Request Graph Transposer with ONNX model
- **Graph Transposer** → **SIT Task Manager**: Response with transposed model
- **SIT Task Manager** → **Holohub**: Trigger Holohub Testing app/service
- **Holohub** → **Holohub**: Convert ONNX to TRT
- **Holohub** → **Holohub**: Calculate IOU Value
- **Holohub** → **SIT Task Manager**: Response with model and testing result

### 5. Steps

1. **Push Model with Metadata**
   - Developer pushes the model along with its metadata to the Huggingface repository.

2. **Trigger System Integration Test**
   - The system detects the new model and triggers the System Integration Test.

3. **Pull Model and Metadata**
   - The SIT Task Manager pulls the latest model and its metadata from the repository.

4. **Trigger Postprocessing Test**
   - The SIT Task Manager sends the model to the Postprocessing Tester to validate its performance.

5. **Return Testing Result**
   - The Postprocessing Tester returns the results of the testing back to the SIT Task Manager.

6. **Request ONNX Conversion**
   - The SIT Task Manager requests the ONNX Converter to convert the model from .pt format to ONNX format.

7. **Convert .pt Model to ONNX**
   - The ONNX Converter processes the model and converts it to ONNX format.

8. **Response with ONNX Model**
   - The ONNX Converter sends the converted ONNX model back to the SIT Task Manager.

9. **Request Graph Transposer**
   - The SIT Task Manager requests the Graph Transposer to optimize the ONNX model.

10. **Response with Transposed Model**
    - The Graph Transposer responds with the optimized (transposed) ONNX model.

11. **Trigger Holohub Testing App/Service**
    - The SIT Task Manager triggers the Holohub to test the model further.

12. **Convert ONNX to TRT**
    - The Holohub converts the ONNX model to TensorRT (TRT) format.

13. **Calculate IOU Value**
    - The Holohub calculates the Intersection over Union (IOU) value to evaluate model performance.

14. **Response with Model and Testing Result**
    - The Holohub sends the final model and testing results back to the SIT Task Manager.