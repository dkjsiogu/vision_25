# 模型标签映射说明

## 概览

本文档说明项目中使用的 YOLO 模型与装甲板类别/颜色/大小的映射关系。当前使用的模型为 **yolov5_0526.xml**（新模型，0526 版本）。

## YOLO 模型输出格式

### 颜色分类（Color）
新模型 yolov5_0526.xml 的颜色类别顺序（从 YOLO 输出第 9-12 位）：

| 索引 | 颜色 | 说明 |
|------|------|------|
| 0 | 红 (Red) | 敌方机器人装甲（通常为红方） |
| 1 | 蓝 (Blue) | 敌方机器人装甲（通常为蓝方） |
| 2 | 灰 (Gray) | 预留/其他 |
| 3 | 紫 (Purple) | 预留/其他 |

### 类别分类（Number ID）
新模型 yolov5_0526.xml 的类别顺序（从 YOLO 输出第 13-21 位）：

| 索引 | 类别代码 | 机器人 | 装甲类型 | 说明 |
|------|---------|--------|---------|------|
| 0 | G | 哨兵 (Sentry) | 小 | 哨兵机器人 |
| 1 | 1 | 英雄 (Hero) | **大** | 英雄机器人（只有大装甲） |
| 2 | 2 | 工程 (Engineer) | 小 | 工程机器人 |
| 3 | 3 | 步兵 (Infantry) | 小 | 步兵机器人 |
| 4 | 4 | 步兵 (Infantry) | 小 | 步兵机器人 |
| 5 | 5 | 步兵 (Infantry) | 小 | 步兵机器人 |
| 6 | O | 前哨站 (Outpost) | 小 | 前哨站（3块小装甲） |
| 7 | Bs | 基地小 (Base Small) | 小 | 基地装甲（小） |
| 8 | Bb | 基地大 (Base Big) | **大** | 基地装甲（大） |

## 代码实现

### [tasks/auto_aim/armor.cpp](tasks/auto_aim/armor.cpp)

**YOLOV5 构造函数**（行 158-182）处理模型输出的标签映射：

```cpp
// 新模型(0526)颜色顺序: 红蓝灰紫 (0=红, 1=蓝, 2=灰, 3=紫)
color = color_id == 0 ? Color::red 
      : color_id == 1 ? Color::blue 
      : color_id == 3 ? Color::purple 
      : Color::extinguish;

// 新模型(0526)类别顺序: G(0), 1-5(1-5), O(6), Bs(7), Bb(8)
name = num_id == 0    ? ArmorName::sentry
     : num_id >= 7    ? ArmorName::base        // Bs(7) 和 Bb(8) 都映射到 base
     : num_id == 6    ? ArmorName::outpost
     : ArmorName(num_id - 1);                  // 1-5 映射到 one-five

// 英雄(1)和基地大(8)是大装甲
type = (num_id == 1 || num_id == 8) ? ArmorType::big : ArmorType::small;
```

**关键映射：**
- 颜色：`color_id` 直接对应 0=红, 1=蓝, 3=紫
- 类别：
  - 0 → `sentry`（哨兵）
  - 1-5 → `one` 到 `five`（英雄/工程/步兵）
  - 6 → `outpost`（前哨站）
  - 7, 8 → `base`（基地）
- 大小：
  - 1（英雄）和 8（基地大）为 **big**
  - 其他均为 **small**

### [tasks/auto_aim/yolos/yolov5.cpp](tasks/auto_aim/yolos/yolov5.cpp)

**check_type 函数**（行 208-215）对装甲类型的合法性检查：

```cpp
// 新模型(0526)支持 Bs(基地小) 和 Bb(基地大)，所以 base 可以是 small 或 big
auto name_ok = (armor.type == ArmorType::small)
                 ? (armor.name != ArmorName::one)      // 英雄只有大装甲
                 : (armor.name != ArmorName::two       // 工程/哨兵/前哨站只有小装甲
                    && armor.name != ArmorName::sentry 
                    && armor.name != ArmorName::outpost);
```

**关键检查：**
- 小装甲：允许所有 except 英雄（1）
- 大装甲：允许英雄（1）、基地（Bs/Bb）；不允许工程（2）、哨兵（0）、前哨站（6）、步兵（3-5）

## 配置与运行

### YAML 配置示例

```yaml
yolo_name: yolov5
yolov5_model_path: assets/yolov5_0526.xml
yolov5_label_schema: "0526"  # 重要：指定标签映射方案
device: CPU
min_confidence: 0.8
use_traditional: true
```

**重要**：`yolov5_label_schema` 参数：
- `"0526"` 或 `"schema_0526"`：使用新模型(0526)的标签映射
- 不设置或其他值：使用 legacy 映射（旧模型兼容）

### 验证方法

1. **检查模型路径**：确保 `yolov5_model_path` 指向正确的 `.xml` 文件
2. **输出日志**：
   ```
   [Tracker] Detected: color=red, name=one, type=big, confidence=0.95
   [Tracker] Detected: color=blue, name=base, type=small, confidence=0.87
   ```
3. **单元测试**：运行 `./build/auto_aim_test` 验证 armor 构造与类型检查

## 故障排查

### 症状：装甲被错误分类

**原因检查清单：**
1. 确认 `yolov5_model_path` 确实是 `yolov5_0526.xml`（不是旧版 `yolov5.xml`）
2. 对比 YOLO 输出第 9-12 位（颜色）与第 13-21 位（类别）的位置是否与模型实际一致
3. 在 `armor.cpp` 的 YOLOV5 构造函数中加 debug 日志，检查 `color_id` 和 `num_id` 的实际值
4. 检查 `check_type()` 是否过滤掉了不期望的类型组合

### 症状：基地装甲无法识别

**原因：** 旧代码假设 base 只有大装甲，新模型支持 Bs(7, small) 和 Bb(8, big)。  
**解决：** 确认代码已更新为：
```cpp
type = (num_id == 1 || num_id == 8) ? ArmorType::big : ArmorType::small;
```
而非旧版本的：
```cpp
type = num_id == 1 ? ArmorType::big : ArmorType::small;  // ❌ 错误
```

## 版本历史

| 版本 | 日期 | 改动 |
|------|------|------|
| v0526 | 2026-01-13 | 新增 Bs/Bb 区分；颜色顺序改为 红蓝灰紫；类别顺序确认为 G/1-5/O/Bs/Bb |
| v之前 | - | 基地只有大装甲；颜色蓝红顺序可能不同 |

---

**注**：如使用其他版本的 YOLO 模型，请根据该模型的实际输出格式对应修改 [armor.cpp](tasks/auto_aim/armor.cpp) 和 [yolov5.cpp](tasks/auto_aim/yolos/yolov5.cpp)。
