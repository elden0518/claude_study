# Git 提交和推送脚本
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Git 提交和推送到 GitHub" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "D:\claude_project\claude_study"

Write-Host "[1/4] 检查当前状态..." -ForegroundColor Yellow
git status
Write-Host ""

Write-Host "[2/4] 添加修改的文件..." -ForegroundColor Yellow
git add langchain_langgraph/02_lc_advanced/11_tools_agents.py
Write-Host ""te

Write-Host "[3/4] 提交修改..." -ForegroundColor Yellow
git commit -m "fix: 修复 11_tools_agents.py 以兼容 LangChain 1.0+

- 移除已弃用的 AgentExecutor 导入
- 使用新的 create_agent API（关键字参数）
- 更新调用方式为 messages 格式
- 更新返回值处理逻辑
- 修复 create_agent 函数签名问题"34
Write-Host ""

Write-Host "[4/4] 推送到 GitHub..." -ForegroundColor Yellow
git push
Write-Host ""

Write-Host "========================================" -ForegroundColor Green
Write-Host "完成！已成功提交并推送到 GitHub" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
