# Common Protocol Buffers

## Overview
Common protocol buffer definitions shared across all VitalStream services.

## Messages

### Core Messages
- **HealthCheckRequest/Response** - Service health checks
- **RequestMetadata** - Request tracking metadata
- **ResponseMetadata** - Response tracking metadata
- **Empty** - Empty message for requests
- **UUID** - UUID wrapper message
- **Coordinate** - Geographic coordinates
- **Address** - Physical address
- **PhoneNumber** - Phone number with type
- **Email** - Email address with type
- **PersonName** - Person name components

### Ranges and Filters
- **TimestampRange** - Time range filtering
- **DurationRange** - Duration range filtering
- **NumericRange** - Numeric range filtering
- **IntegerRange** - Integer range filtering
- **SearchQuery** - Advanced search parameters
- **FilterOption** - Individual filter criteria
- **SearchFilter** - Combined filter logic
- **SortOption** - Sorting parameters

### Configuration
- **Configuration** - Configuration key-value pairs
- **FeatureFlag** - Feature toggle configuration
- **ValidationRule** - Validation rule definition
- **Tag** - Key-value tag
- **Label** - Kubernetes-style label
- **Annotation** - Metadata annotation

### Monitoring
- **MetricValue** - Metric data point
- **LogEntry** - Structured log entry
- **VersionInfo** - Version information
- **EnvironmentInfo** - Environment details
- **ServiceInfo** - Service metadata

## Enums

### HealthCheckResponse.ServingStatus
- `SERVING_STATUS_UNSPECIFIED` (0)
- `SERVING_STATUS_SERVING` (1)
- `SERVING_STATUS_NOT_SERVING` (2)
- `SERVING_STATUS_SERVICE_UNKNOWN` (3)

### LogEntry.LogLevel
- `LOG_LEVEL_UNSPECIFIED` (0)
- `LOG_LEVEL_DEBUG` (1)
- `LOG_LEVEL_INFO` (2)
- `LOG_LEVEL_WARNING` (3)
- `LOG_LEVEL_ERROR` (4)
- `LOG_LEVEL_FATAL` (5)

### SortOption.Direction
- `DIRECTION_UNSPECIFIED` (0)
- `DIRECTION_ASC` (1)
- `DIRECTION_DESC` (2)

## Usage Examples

### Health Check
```python
request = HealthCheckRequest(service="alarm-service")
response = client.HealthCheck(request)
```

### Search Query
```python
query = SearchQuery(
    text="heart rate",
    fields=["description", "metadata"],
    filter=FilterOption(
        field="severity",
        operator=FilterOption.Operator.OPERATOR_GREATER_THAN,
        values=["2"]
    ),
    sort=SortOption(field="created_at", direction=SortOption.Direction.DESC)
)
```

### Metadata
```python
metadata = RequestMetadata(
    request_id="req-123",
    user_id="user-456",
    client_version="1.0.0"
)
```

## Best Practices

1. **Health Checks**
   - Implement for all services
   - Include service name in request
   - Return appropriate serving status

2. **Metadata**
   - Include request IDs for tracing
   - Add user and session information
   - Track client versions

3. **Pagination**
   - Use consistent pagination patterns
   - Include total counts when possible
   - Support cursor-based pagination

4. **Error Handling**
   - Use standardized error codes
   - Include detailed error messages
   - Provide context for debugging
