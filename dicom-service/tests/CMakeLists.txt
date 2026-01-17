# Basic tests using CTest
# Use target file to ensure we run the built executable
add_test(NAME dicom_service_run COMMAND $<TARGET_FILE:dicom_service>)
set_tests_properties(dicom_service_run PROPERTIES PASS_REGULAR_EXPRESSION "started|stub")

# Optional: provide a custom target to run the service locally
add_custom_target(run_dicom_service
    COMMAND $<TARGET_FILE:dicom_service>
    DEPENDS dicom_service
    COMMENT "Run dicom_service (stub)"
)
