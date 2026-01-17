#include <gtest/gtest.h>
#include <iostream>

// Simple test to verify Google Test works
TEST(SimpleTest, BasicAssertions) {
    EXPECT_EQ(1, 1);
    EXPECT_TRUE(true);
    EXPECT_FALSE(false);
}

int main(int argc, char **argv) {
    std::cout << "🧪 Running Simple Test..." << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
