#ifndef VITALSTREAM_GRPC_SERVER_H
#define VITALSTREAM_GRPC_SERVER_H

namespace vitalstream {
namespace dicom {

class GRPCServer {
public:
    GRPCServer();
    void start();
    void stop();
};

} // namespace dicom
} // namespace vitalstream

#endif // VITALSTREAM_GRPC_SERVER_H
