# Alarm Service Protocol Buffers

## Overview

The Alarm Service provides real-time alarm management for VitalStream medical monitoring systems.

## Messages

### Core Messages

- **Alarm** - Main alarm entity with all metadata
- **AlarmEvent** - Alarm events for streaming
- **AlarmStatistics** - Aggregated alarm statistics

### Request/Response Pairs

- **CreateAlarmRequest/Response** - Create new alarms
- **GetAlarmsRequest/Response** - Query alarms with filtering
- **AcknowledgeAlarmRequest/Response** - Acknowledge active alarms
- **ResolveAlarmRequest/Response** - Resolve alarms
- **SilenceAlarmRequest/Response** - Temporarily silence alarms
- **EscalateAlarmRequest/Response** - Escalate alarm priority

### Streaming

- **StreamAlarmsRequest** - Filter criteria for alarm streaming
- **AlarmEvent** - Bidirectional alarm events
- **SubscribeAlarmsRequest/Response** - Subscribe to notifications

### Batch Operations

- **BatchAlarmRequest/Response** - Execute multiple operations
- **BatchAlarmResult** - Individual operation results

## Enums

### AlarmSeverity

- `ALARM_SEVERITY_UNSPECIFIED` (0)
- `ALARM_SEVERITY_INFO` (1)
- `ALARM_SEVERITY_LOW` (2)
- `ALARM_SEVERITY_MEDIUM` (3)
- `ALARM_SEVERITY_HIGH` (4)
- `ALARM_SEVERITY_CRITICAL` (5)

### AlarmStatus

- `ALARM_STATUS_UNSPECIFIED` (0)
- `ALARM_STATUS_ACTIVE` (1)
- `ALARM_STATUS_ACKNOWLEDGED` (2)
- `ALARM_STATUS_RESOLVED` (3)
- `ALARM_STATUS_SILENCED` (4)
- `ALARM_STATUS_ESCALATED` (5)

### AlarmType

- `ALARM_TYPE_UNSPECIFIED` (0)
- `ALARM_TYPE_ARRHYTHMIA` (1)
- `ALARM_TYPE_TACHYCARDIA` (2)
- `ALARM_TYPE_BRADYCARDIA` (3)
- `ALARM_TYPE_AFIB` (4)
- `ALARM_TYPE_VFIB` (5)
- `ALARM_TYPE_PVC` (6)
- `ALARM_TYPE_ST_ELEVATION` (7)
- `ALARM_TYPE_ST_DEPRESSION` (8)
- `ALARM_TYPE_LEAD_OFF` (9)
- `ALARM_TYPE_SIGNAL_QUALITY` (10)
- `ALARM_TYPE_DEVICE_ERROR` (11)
- `ALARM_TYPE_BATTERY_LOW` (12)
- `ALARM_TYPE_DISCONNECTED` (13)
- `ALARM_TYPE_THRESHOLD_BREACH` (14)
- `ALARM_TYPE_SYSTEM_ERROR` (15)
- `ALARM_TYPE_CUSTOM` (99)

### NotificationChannel

- `NOTIFICATION_CHANNEL_UNSPECIFIED` (0)
- `NOTIFICATION_CHANNEL_EMAIL` (1)
- `NOTIFICATION_CHANNEL_SMS` (2)
- `NOTIFICATION_CHANNEL_PAGER` (3)
- `NOTIFICATION_CHANNEL_SLACK` (4)
- `NOTIFICATION_CHANNEL_WEBHOOK` (5)
- `NOTIFICATION_CHANNEL_MOBILE_PUSH` (6)
- `NOTIFICATION_CHANNEL_IN_APP` (7)

## Service RPCs

### Core Operations

- `CreateAlarm` - Create new alarm
- `GetAlarms` - Query alarms with pagination
- `AcknowledgeAlarm` - Acknowledge alarm
- `ResolveAlarm` - Resolve alarm
- `SilenceAlarm` - Temporarily silence alarm
- `EscalateAlarm` - Escalate alarm priority

### -Streaming

- `StreamAlarms` - Server-side alarm streaming
- `AlarmChannel` - Bidirectional alarm communication
- `SubscribeAlarms` - Subscribe to notifications
- `UnsubscribeAlarms` - Unsubscribe from notifications

### Advanced Features

- `GetStatistics` - Get alarm statistics
- `BatchAlarmOperations` - Execute batch operations
- `HealthCheck` - Service health check

## Usage Examples

### Create Alarm

```python
request = CreateAlarmRequest(
    patient_id="patient-123",
    type=AlarmType.ALARM_TYPE_TACHYCARDIA,
    severity=AlarmSeverity.ALARM_SEVERITY_HIGH,
    message="Heart rate exceeded threshold",
    metadata={"heart_rate": "145", "threshold": "120"}
)
```

### Stream Alarms

```python
request = StreamAlarmsRequest(
    patient_id="patient-123",
    min_severity=[AlarmSeverity.ALARM_SEVERITY_MEDIUM]
)
```

## Best Practices

1. **Alarm Creation**
   - Always include patient_id for traceability
   - Use appropriate severity levels
   - Include relevant metadata for context

2. **Acknowledgment**
   - Always include acknowledged_by for audit trail
   - Add notes for context
   - Consider escalation if needed

3. **Streaming**
   - Use filters to reduce bandwidth
   - Handle connection failures gracefully
   - Implement reconnection logic

4. **Error Handling**
   - Check error responses
   - Implement retry logic
   - Log failures appropriately
