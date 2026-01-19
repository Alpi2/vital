package com.vitalstream.hl7.messaging;

import ca.uhn.hl7v2.HL7Exception;
import ca.uhn.hl7v2.model.v25.message.ORM_O01;
import ca.uhn.hl7v2.model.v25.segment.*;
import ca.uhn.hl7v2.model.v25.group.ORM_O01_ORDER;
import ca.uhn.hl7v2.model.v25.datatype.XCN;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * ORM (Order) Message Handler
 * 
 * Handles HL7 ORM^O01 messages for:
 * - Laboratory orders
 * - Radiology orders
 * - Medication orders
 * - Procedure orders
 * 
 * @author VitalStream Development Team
 * @version 1.0.0
 * @since 2026-01-03
 */
@Component
public class ORMMessageHandler {

    private static final Logger log = LoggerFactory.getLogger(ORMMessageHandler.class);

    /**
     * Order information
     */
    public static class OrderInfo {
        private String orderId;
        private String orderCode;
        private String orderName;
        private String orderType;          // LAB, RAD, MED, PROC
        private String priority;           // S=Stat, A=ASAP, R=Routine
        private String orderDateTime;
        private String orderingProvider;
        private String orderStatus;        // NW=New, IP=In Progress, CM=Complete, CA=Cancelled
        private Map<String, String> orderDetails;
        
        // Getters and setters
        public String getOrderId() { return orderId; }
        public void setOrderId(String orderId) { this.orderId = orderId; }
        
        public String getOrderCode() { return orderCode; }
        public void setOrderCode(String orderCode) { this.orderCode = orderCode; }
        
        public String getOrderName() { return orderName; }
        public void setOrderName(String orderName) { this.orderName = orderName; }
        
        public String getOrderType() { return orderType; }
        public void setOrderType(String orderType) { this.orderType = orderType; }
        
        public String getPriority() { return priority; }
        public void setPriority(String priority) { this.priority = priority; }
        
        public String getOrderDateTime() { return orderDateTime; }
        public void setOrderDateTime(String orderDateTime) { this.orderDateTime = orderDateTime; }
        
        public String getOrderingProvider() { return orderingProvider; }
        public void setOrderingProvider(String orderingProvider) { 
            this.orderingProvider = orderingProvider; 
        }
        
        public String getOrderStatus() { return orderStatus; }
        public void setOrderStatus(String orderStatus) { this.orderStatus = orderStatus; }
        
        public Map<String, String> getOrderDetails() { return orderDetails; }
        public void setOrderDetails(Map<String, String> orderDetails) { 
            this.orderDetails = orderDetails; 
        }
        
        public boolean isStat() {
            return "S".equals(priority);
        }
    }

    /**
     * Parse ORM^O01 message
     */
    public Map<String, Object> parseORMMessage(ORM_O01 message) throws HL7Exception {
        Map<String, Object> result = new HashMap<>();
        
        // Parse MSH
        MSH msh = message.getMSH();
        result.put("messageId", msh.getMessageControlID().getValue());
        result.put("timestamp", msh.getDateTimeOfMessage().getTime().getValue());
        
        // Parse PID
        PID pid = message.getPATIENT().getPID();
        result.put("patientId", pid.getPatientID().getIDNumber().getValue());
        result.put("mrn", pid.getPatientIdentifierList(0).getIDNumber().getValue());
        
        String lastName = pid.getPatientName(0).getFamilyName().getSurname().getValue();
        String firstName = pid.getPatientName(0).getGivenName().getValue();
        result.put("patientName", firstName + " " + lastName);
        
        // Parse orders
        List<OrderInfo> orders = new ArrayList<>();
        
        int orderCount = message.getORDERReps();
        for (int i = 0; i < orderCount; i++) {
            ORM_O01_ORDER order = message.getORDER(i);
            
            // Parse ORC (Common Order)
            ORC orc = order.getORC();
            OrderInfo orderInfo = new OrderInfo();
            
            orderInfo.setOrderId(orc.getPlacerOrderNumber().getEntityIdentifier().getValue());
            orderInfo.setOrderStatus(orc.getOrderStatus().getValue());
            
            // Ordering provider
            if (orc.getOrderingProvider().length > 0) {
                XCN provider = orc.getOrderingProvider(0);
                String providerName = provider.getGivenName().getValue() + " " +
                                    provider.getFamilyName().getSurname().getValue();
                orderInfo.setOrderingProvider(providerName);
            }
            
            // Parse OBR (Observation Request) - for lab/rad orders
            if (order.getORDER_DETAIL() != null && 
                order.getORDER_DETAIL().getOBR() != null) {
                
                OBR obr = order.getORDER_DETAIL().getOBR();
                
                orderInfo.setOrderCode(obr.getUniversalServiceIdentifier().getIdentifier().getValue());
                orderInfo.setOrderName(obr.getUniversalServiceIdentifier().getText().getValue());
                orderInfo.setPriority(obr.getPriorityOBR().getValue());
                orderInfo.setOrderDateTime(obr.getObservationDateTime().getTime().getValue());
                
                // Determine order type from code
                String code = orderInfo.getOrderCode();
                if (code.startsWith("LAB")) {
                    orderInfo.setOrderType("LAB");
                } else if (code.startsWith("RAD")) {
                    orderInfo.setOrderType("RAD");
                } else {
                    orderInfo.setOrderType("OTHER");
                }
            }
            
            orders.add(orderInfo);
            
            // Log STAT orders
            if (orderInfo.isStat()) {
                log.warn("STAT ORDER: {} for patient {}",
                        orderInfo.getOrderName(),
                        result.get("mrn"));
            }
        }
        
        result.put("orders", orders);
        result.put("orderCount", orders.size());
        
        log.info("Parsed ORM^O01 message: {} orders for patient {}",
                orders.size(), result.get("mrn"));
        
        return result;
    }

    /**
     * Create ORM^O01 message for lab order
     */
    public String createLabOrder(String patientId, String patientName, 
                                String testCode, String testName, 
                                String priority) {
        StringBuilder hl7 = new StringBuilder();
        String timestamp = getCurrentTimestamp();
        String orderId = "ORD" + System.currentTimeMillis();
        
        // MSH segment
        hl7.append("MSH|^~\\&|VitalStream|Hospital|LIS|Hospital|")
           .append(timestamp)
           .append("||");
        hl7.append("ORM^O01|MSG").append(System.currentTimeMillis()).append("|P|2.5\r");
        
        // PID segment
        hl7.append("PID|1||").append(patientId).append("|||");
        hl7.append(patientName).append("\r");
        
        // ORC segment
        hl7.append("ORC|NW|").append(orderId).append("|||");
        hl7.append("NW|").append(priority).append("\r");
        
        // OBR segment
        hl7.append("OBR|1|").append(orderId).append("|||");
        hl7.append(testCode).append("^").append(testName).append("|");
        hl7.append(priority).append("|").append(timestamp).append("\r");
        
        return hl7.toString();
    }

    private String getCurrentTimestamp() {
        return new java.text.SimpleDateFormat("yyyyMMddHHmmss").format(new Date());
    }
}
