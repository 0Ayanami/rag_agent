---
name: service_request
description: 将无法直接解决的问题整理为服务工单草稿
version: "1.0.0"
tools:
  - draft_service_request
  - get_request_identity
---
当现有信息不足以解决问题，或用户明确要求形成工单时：

1. 先确认事项标题、背景、期望结果和紧急程度。
2. 调用 `draft_service_request` 生成草稿。
3. 明确说明草稿没有被实际提交，必须由用户人工确认。
4. 不得声称已通知部门、已创建工单号或已完成审批。
