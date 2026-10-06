# Netlify 部署说明

## 目录结构

- `app/frontend/`：静态前端页面
- `netlify/functions/api.mjs`：Netlify Function，提供 `/api/*` 接口
- `netlify/functions/data.mjs`：由论文脱敏数据生成的静态数据模块
- `data/processed/`：论文数据快照
- `app/backend/`：本地 FastAPI 版本，便于后续本地开发和算法服务拆分

## Netlify 配置

项目根目录的 `netlify.toml` 已配置：

- 发布目录：`app/frontend`
- Functions 目录：`netlify/functions`
- `/api/*` 重写到 `/.netlify/functions/api/*`
- 单页应用路由回退到 `/index.html`

## 部署方式

### 方式一：Netlify 连接 GitHub（推荐）

1. 将本目录上传到 GitHub 仓库。
2. 在 Netlify 中选择 **Add new site → Import an existing project**。
3. 选择 GitHub 仓库和对应分支。
4. 构建命令可留空，发布目录填写 `app/frontend`；如果读取 `netlify.toml`，可直接使用默认配置。
5. 点击部署。

### 方式二：Netlify CLI

在项目根目录执行：

```text
npm install -g netlify-cli
netlify login
netlify init
netlify deploy --prod
```

## 本地检查

本地静态页面可以直接使用任意静态服务器打开，但完整 API 需要使用 Netlify Dev 或本地 FastAPI：

```text
netlify dev
```

启动后访问 Netlify Dev 输出的本地地址。前端仍然通过 `/api/*` 访问后端函数。

## 数据与隐私说明

部署版本使用的是论文案例标签、风险预测明细、SHAP 示例因素和 EVT 示例规则的脱敏快照。原始事故报告、图片、绝对路径、个人联系方式和未脱敏文档不应提交到公开仓库。

当前大语言模型模块使用“知识增强 Agent 演示适配器”，未配置外部模型密钥时不会调用第三方服务。接入真实模型时，密钥应放在 Netlify Environment Variables 中，不能写入前端代码或 Git 仓库。
