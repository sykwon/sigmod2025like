# compiler
CC = g++

# C++ compiler option
CXXFLAGS = -Wall -O2
CXXFLAGS_DBG = -Wall -fno-inline-functions

# linker option
LDFLAGS = `pkg-config --libs --cflags icu-uc icu-io re2`
LDFLAGS_DBG = `pkg-config --libs --cflags icu-uc icu-io re2` -DDEBUG -g

# source directory
SRC_DIR = ./cpp

# object directory
OBJ_DIR = ./obj
OBJ_DIR_DBG = ./obj/debug

# exe file name
TARGET = main
TARGET_DBG = main_debug

# source files to make
SRCS = $(notdir $(wildcard $(SRC_DIR)/*.cpp))

OBJS = $(SRCS:.cpp=.o)

OBJECTS = $(patsubst %.o,$(OBJ_DIR)/%.o,$(OBJS))
DEPS = $(OBJECTS:.o=.d)

OBJECTS_DBG = $(patsubst %.o,$(OBJ_DIR_DBG)/%.o,$(OBJS))
DEPS_DBG = $(OBJECTS_DBG:.o=.d)

all: $(TARGET)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	@$(CC) $(CXXFLAGS) -c $< -o $@ -MD $(LDFLAGS)

$(TARGET) : $(OBJECTS)
	$(CC) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

debug: $(TARGET_DBG)

$(OBJ_DIR_DBG)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	@$(CC) $(CXXFLAGS_DBG) -c $< -o $@ -MD $(LDFLAGS_DBG)

$(TARGET_DBG) : $(OBJECTS_DBG)
	$(CC) $(CXXFLAGS_DBG) $(OBJECTS_DBG) -o $(TARGET_DBG) $(LDFLAGS_DBG)

.PHONY: clean all
clean:
	rm -f $(OBJECTS) $(DEPS) $(TARGET) $(OBJECTS_DBG) $(DEPS_DBG) $(TARGET_DBG)

-include $(DEPS)
