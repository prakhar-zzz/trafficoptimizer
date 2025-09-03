package com.prakhar.trafficoptimizer.controller;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

@CrossOrigin(origins = "*")
@RestController
@RequestMapping("/api/traffic")
public class TrafficController {

    private static final int BUFFER_TIME = 5;
    private static final int MIN_GREEN_TIME = 20;
    private static final int MAX_GREEN_TIME = 70;

    private static final Map<String, Double> TIME_PER_VEHICLE = Map.of(
            "car", 1.5,
            "motorbike", 1.0,
            "bus", 2.5,
            "truck", 2.5
    );

    private Map<String, Integer> lastSignalTimings = new HashMap<>();

    @PostMapping("/update-counts")
    public ResponseEntity<Map<String, Integer>> updateVehicleCounts(
            @RequestBody Map<String, Map<String, Integer>> laneClassCounts) {

        Map<String, Integer> signalTimings = calculateSignalTimings(laneClassCounts);
        lastSignalTimings = signalTimings;
        return ResponseEntity.ok(signalTimings);
    }

    @GetMapping("/last-green")
    public ResponseEntity<Map<String, Integer>> getLastGreenSignalTimings() {
        return ResponseEntity.ok(lastSignalTimings);
    }

    @PostMapping("/run-yolo")
    public ResponseEntity<Map<String, Integer>> runYoloAndGetTimings() {
        try {
            ProcessBuilder pb = new ProcessBuilder("python", "C:\\prakhar\\trafficoptimizer\\yolo_vision\\yolo_vehicle_counter.py");
            pb.directory(new File("C:\\prakhar\\trafficoptimizer\\yolo_vision"));
            pb.redirectErrorStream(true);
            Process process = pb.start();

            // 2. Wait for completion
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                return ResponseEntity.status(500).body(null);
            }

            File jsonFile = new File("C:\\prakhar\\trafficoptimizer\\yolo_vision\\output.json");
            ObjectMapper mapper = new ObjectMapper();
            TypeReference<Map<String, Map<String, Integer>>> typeRef = new TypeReference<>() {};
            Map<String, Map<String, Integer>> laneClassCounts = mapper.readValue(jsonFile, typeRef);

            Map<String, Integer> signalTimings = calculateSignalTimings(laneClassCounts);
            lastSignalTimings = signalTimings;
            return ResponseEntity.ok(signalTimings);

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body(null);
        }
    }



    @GetMapping("/yolo-timings")
    public ResponseEntity<Map<String, Integer>> getYoloSignalTimings() {
        try {
            String outputPath = "C:\\prakhar\\trafficoptimizer\\yolo_vision\\output.json";
            ObjectMapper mapper = new ObjectMapper();
            TypeReference<Map<String, Map<String, Integer>>> typeRef = new TypeReference<>() {};
            Map<String, Map<String, Integer>> laneClassCounts = mapper.readValue(new File(outputPath), typeRef);

            Map<String, Integer> signalTimings = calculateSignalTimings(laneClassCounts);
            lastSignalTimings = signalTimings;

            return ResponseEntity.ok(signalTimings);
        } catch (Exception e) {
            return ResponseEntity.status(500).body(null);
        }
    }

    private Map<String, Integer> calculateSignalTimings(Map<String, Map<String, Integer>> laneClassCounts) {
        Map<String, Integer> signalTimings = new HashMap<>();

        for (Map.Entry<String, Map<String, Integer>> laneEntry : laneClassCounts.entrySet()) {
            String lane = laneEntry.getKey();
            Map<String, Integer> classCounts = laneEntry.getValue();

            double totalTime = 0;
            for (Map.Entry<String, Integer> classEntry : classCounts.entrySet()) {
                String vehicleType = classEntry.getKey();
                int count = classEntry.getValue();
                double timePerVehicle = TIME_PER_VEHICLE.getOrDefault(vehicleType, 2.0);
                totalTime += count * timePerVehicle;
            }

            totalTime += BUFFER_TIME;
            int greenTime = (int) Math.ceil(totalTime);
            greenTime = Math.max(MIN_GREEN_TIME, Math.min(MAX_GREEN_TIME, greenTime));
            signalTimings.put(lane, greenTime);
        }

        return signalTimings;
    }
}
