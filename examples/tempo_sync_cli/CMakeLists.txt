# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Offline test harness for host tempo / downbeat sync (tempo-sync fork):
# engine + TempoSyncController against a simulated DAW transport.

add_executable(tempo_sync_cli main.cpp)
target_link_libraries(tempo_sync_cli PRIVATE magentart::core)

# C++ beat tracker vs. the Python reference (rhythm/au_sync/scripts/check_tracker_port.py)
add_executable(tempo_tracker_test tempo_tracker_test.cpp)
target_link_libraries(tempo_tracker_test PRIVATE magentart::core)
