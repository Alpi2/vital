package com.vitalstream.hl7.messaging;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CompletableFuture;

/**
 * Bidirectional Data Exchange Service
 * 
 * Manages bidirectional communication between VitalStream and external systems:
 * - Send data to external systems (EHR, LIS, PACS)
 * - Receive data from external systems
 * - Request/response pattern
 * - Asynchronous messaging
 * - Message correlation
 * 
 * Uses RabbitMQ for reliable message delivery.
 * 
 * @author VitalStream Development Team
 * @version 1.0.0
 * @since 2026-01-03
 */
@Service
public class BidirectionalExchangeService {

    private static final Logger log = LoggerFactory.getLogger(BidirectionalExchangeService.class);

    @Autowired
    private RabbitTemplate rabbitTemplate;

    // Message correlation map: messageId -> CompletableFuture
    private final Map<String, CompletableFuture<Map<String, Object>>> pendingRequests = 
        new ConcurrentHashMap<>();

    // Exchange and queue names
    private static final String OUTBOUND_EXCHANGE = "vitalstream.outbound";
    @SuppressWarnings("unused") // Reserved for future RabbitMQ listener implementation
    private static final String INBOUND_EXCHANGE = "vitalstream.inbound";
    @SuppressWarnings("unused") // Reserved for future RabbitMQ listener implementation
    private static final String RESPONSE_QUEUE = "vitalstream.responses";

    /**
     * Send data to external system and wait for response
     * 
     * @param destination Target system (EHR, LIS, PACS)
     * @param messageType Message type (ADT, ORU, ORM, etc.)
     * @param data Message data
     * @return Response from external system
     */
    public CompletableFuture<Map<String, Object>> sendAndWaitForResponse(
            String destination, String messageType, Map<String, Object> data) {
        
        String messageId = UUID.randomUUID().toString();
        CompletableFuture<Map<String, Object>> future = new CompletableFuture<>();
        
        // Store future for correlation
        pendingRequests.put(messageId, future);
        
        // Build message
        Map<String, Object> message = new HashMap<>();
        message.put("messageId", messageId);
        message.put("destination", destination);
        message.put("messageType", messageType);
        message.put("timestamp", new Date());
        message.put("data", data);
        
        // Send to outbound exchange
        String routingKey = destination + "." + messageType;
        rabbitTemplate.convertAndSend(OUTBOUND_EXCHANGE, routingKey, message);
        
        log.info("Sent {} message to {} with ID: {}", messageType, destination, messageId);
        
        // Set timeout (30 seconds)
        CompletableFuture.delayedExecutor(30, java.util.concurrent.TimeUnit.SECONDS)
            .execute(() -> {
                if (!future.isDone()) {
                    future.completeExceptionally(
                        new java.util.concurrent.TimeoutException(
                            "No response received within 30 seconds"));
                    pendingRequests.remove(messageId);
                }
            });
        
        return future;
    }

    /**
     * Send data without waiting for response (fire-and-forget)
     */
    public void sendAsync(String destination, String messageType, Map<String, Object> data) {
        String messageId = UUID.randomUUID().toString();
        
        Map<String, Object> message = new HashMap<>();
        message.put("messageId", messageId);
        message.put("destination", destination);
        message.put("messageType", messageType);
        message.put("timestamp", new Date());
        message.put("data", data);
        
        String routingKey = destination + "." + messageType;
        rabbitTemplate.convertAndSend(OUTBOUND_EXCHANGE, routingKey, message);
        
        log.info("Sent async {} message to {}", messageType, destination);
    }

    /**
     * Handle response from external system
     * 
     * This method should be called by a RabbitMQ listener
     */
    public void handleResponse(Map<String, Object> response) {
        String correlationId = (String) response.get("correlationId");
        
        if (correlationId == null) {
            log.warn("Received response without correlation ID");
            return;
        }
        
        CompletableFuture<Map<String, Object>> future = pendingRequests.remove(correlationId);
        
        if (future != null) {
            future.complete(response);
            log.info("Completed request with correlation ID: {}", correlationId);
        } else {
            log.warn("Received response for unknown correlation ID: {}", correlationId);
        }
    }

    /**
     * Send vital signs to EHR
     */
    public CompletableFuture<Map<String, Object>> sendVitalSignsToEHR(
            String ehrSystem, String patientId, Map<String, Object> vitalSigns) {
        
        Map<String, Object> data = new HashMap<>();
        data.put("patientId", patientId);
        data.put("vitalSigns", vitalSigns);
        data.put("timestamp", new Date());
        
        return sendAndWaitForResponse(ehrSystem, "ORU", data);
    }

    /**
     * Request patient data from EHR
     */
    public CompletableFuture<Map<String, Object>> requestPatientData(
            String ehrSystem, String patientId) {
        
        Map<String, Object> data = new HashMap<>();
        data.put("patientId", patientId);
        data.put("requestType", "PATIENT_QUERY");
        
        return sendAndWaitForResponse(ehrSystem, "QRY", data);
    }

    /**
     * Send lab order to LIS
     */
    public CompletableFuture<Map<String, Object>> sendLabOrder(
            String patientId, String testCode, String priority) {
        
        Map<String, Object> data = new HashMap<>();
        data.put("patientId", patientId);
        data.put("testCode", testCode);
        data.put("priority", priority);
        data.put("orderDateTime", new Date());
        
        return sendAndWaitForResponse("LIS", "ORM", data);
    }

    /**
     * Send ECG waveform to PACS
     */
    public void sendECGToPACS(String patientId, byte[] dicomData) {
        Map<String, Object> data = new HashMap<>();
        data.put("patientId", patientId);
        data.put("studyType", "ECG");
        data.put("dicomData", Base64.getEncoder().encodeToString(dicomData));
        
        sendAsync("PACS", "DICOM_STORE", data);
    }

    /**
     * Get pending request count
     */
    public int getPendingRequestCount() {
        return pendingRequests.size();
    }

    /**
     * Clear all pending requests (for shutdown)
     */
    public void clearPendingRequests() {
        pendingRequests.forEach((id, future) -> {
            future.completeExceptionally(
                new Exception("Service shutting down"));
        });
        pendingRequests.clear();
        log.info("Cleared all pending requests");
    }
}
