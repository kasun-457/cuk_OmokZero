"""GUI interface for Human vs AlphaZero AI (Tkinter)."""
import threading
import tkinter as tk
from tkinter import messagebox

from mcts import MonteCarloTreeSearch, TreeNode
from config import CFG
from gomoku.gomoku_game import GomokuGame, BOARD_SIZE

# ── 보드 시각화 상수 ──────────────────────────────────────────────
CELL      = 52          # 격자 간격 (px)
MARGIN    = 40          # 보드 여백
STONE_R   = 20          # 돌 반지름
BOARD_PX  = MARGIN * 2 + CELL * (BOARD_SIZE - 1)

COLOR_BG      = "#DCB85A"   # 바둑판 배경 (황금색)
COLOR_LINE    = "#8B6914"
COLOR_BLACK   = "#111111"
COLOR_WHITE   = "#F5F5F5"
COLOR_W_EDGE  = "#999999"
COLOR_LAST    = "#FF3333"   # 마지막 착수 표시
COLOR_STATUS  = "#2C2C2C"


class GomokuGUI:
    def __init__(self, net, mcts_sims=CFG.num_mcts_sims):
        self.net  = net
        self.sims = mcts_sims

        self.root = tk.Tk()
        self.root.title("오목 — Human vs AlphaZero")
        self.root.resizable(False, False)

        self._build_ui()
        self._new_game(human_first=True)   # 첫 화면은 흑(먼저) 선택

    # ── UI 구성 ──────────────────────────────────────────────────
    def _build_ui(self):
        # 상단 상태 레이블
        self.status_var = tk.StringVar(value="")
        lbl = tk.Label(self.root, textvariable=self.status_var,
                       font=("Malgun Gothic", 14, "bold"),
                       bg="#3A3A3A", fg="white", pady=8)
        lbl.pack(fill=tk.X)

        # 캔버스
        self.canvas = tk.Canvas(self.root,
                                width=BOARD_PX, height=BOARD_PX,
                                bg=COLOR_BG, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_click)

        # 하단 버튼 바
        btn_frame = tk.Frame(self.root, bg="#3A3A3A", pady=6)
        btn_frame.pack(fill=tk.X)

        style = dict(font=("Malgun Gothic", 11), padx=14, pady=4,
                     relief=tk.FLAT, cursor="hand2")

        tk.Button(btn_frame, text="내가 먼저 (흑)",
                  bg="#4A90D9", fg="white",
                  command=lambda: self._new_game(human_first=True),
                  **style).pack(side=tk.LEFT, padx=8)

        tk.Button(btn_frame, text="AI가 먼저 (백)",
                  bg="#E87040", fg="white",
                  command=lambda: self._new_game(human_first=False),
                  **style).pack(side=tk.LEFT, padx=4)

        tk.Button(btn_frame, text="다시 시작",
                  bg="#5C5C5C", fg="white",
                  command=self._restart,
                  **style).pack(side=tk.RIGHT, padx=8)

    # ── 게임 초기화 ───────────────────────────────────────────────
    def _new_game(self, human_first: bool):
        self.game         = GomokuGame()
        self.mcts         = MonteCarloTreeSearch(self.net)
        self.node         = TreeNode()
        self.human_player = 1 if human_first else -1
        self.game_over    = False
        self.ai_thinking  = False
        self.last_move    = None

        sym = "흑(X)" if human_first else "백(O)"
        self._set_status(f"당신: {sym}  |  클릭으로 착수하세요")
        self._draw_board()

        # AI가 먼저인 경우 즉시 착수
        if not human_first:
            self.root.after(300, self._ai_move)

    def _restart(self):
        self._new_game(human_first=(self.human_player == 1))

    # ── 클릭 이벤트 ──────────────────────────────────────────────
    def _on_click(self, event):
        if self.game_over or self.ai_thinking:
            return
        if self.game.current_player != self.human_player:
            return

        # 픽셀 → 격자 좌표
        col = round((event.x - MARGIN) / CELL)
        row = round((event.y - MARGIN) / CELL)

        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return

        action = row * BOARD_SIZE + col
        if self.game.get_valid_moves()[action] == 0:
            self._set_status("⚠  이미 돌이 놓인 자리입니다.")
            return

        self._place_stone(action)

    # ── 착수 처리 ─────────────────────────────────────────────────
    def _place_stone(self, action):
        self.game.play_action(action)
        self.node         = self._advance_node(self.node, action)
        self.last_move    = action
        self._draw_board()

        over, value = self.game.check_game_over()
        if over:
            self._end_game(value)
            return

        # AI 차례
        self.root.after(50, self._ai_move)

    def _ai_move(self):
        if self.game_over:
            return
        self.ai_thinking = True
        self._set_status("AI 생각 중... ⏳")

        def run():
            best = self.mcts.search(self.game, self.node, CFG.temp_final)
            self.root.after(0, lambda: self._apply_ai_move(best))

        threading.Thread(target=run, daemon=True).start()

    def _apply_ai_move(self, best_child):
        self.ai_thinking = False
        action = best_child.action
        r, c   = divmod(action, BOARD_SIZE)

        self.game.play_action(action)
        self.node      = self._advance_node(self.node, action)
        self.last_move = action
        self._draw_board()
        self._set_status(f"AI 착수: ({r}, {c})  |  당신의 차례입니다")

        over, value = self.game.check_game_over()
        if over:
            self._end_game(value)

    def _advance_node(self, node, action):
        """트리 재사용: 이미 확장된 노드라면 해당 자식을 반환."""
        for child in node.children:
            if child.action == action:
                child.parent = None
                return child
        new_node        = TreeNode()
        new_node.action = action
        return new_node

    # ── 게임 종료 ─────────────────────────────────────────────────
    def _end_game(self, value):
        self.game_over = True
        self._draw_board()

        # value는 마지막으로 돌을 놓은 플레이어 관점
        # game.current_player 는 이미 교대된 상태
        last_player = -self.game.current_player
        if value == 0:
            msg = "무승부!"
        elif last_player == self.human_player:
            msg = "🎉 축하합니다! 당신이 이겼습니다!"
        else:
            msg = "💻 AI가 이겼습니다. 다시 도전하세요!"

        self._set_status(msg)
        self.root.after(400, lambda: messagebox.showinfo("게임 종료", msg))

    # ── 보드 그리기 ───────────────────────────────────────────────
    def _draw_board(self):
        c = self.canvas
        c.delete("all")

        # 격자
        for i in range(BOARD_SIZE):
            x = MARGIN + i * CELL
            y = MARGIN + i * CELL
            c.create_line(MARGIN, y, MARGIN + (BOARD_SIZE-1)*CELL, y,
                          fill=COLOR_LINE, width=1)
            c.create_line(x, MARGIN, x, MARGIN + (BOARD_SIZE-1)*CELL,
                          fill=COLOR_LINE, width=1)
            # 좌표 레이블
            c.create_text(MARGIN + i*CELL, MARGIN//2,
                          text=str(i), font=("Consolas", 9), fill="#5A3E1B")
            c.create_text(MARGIN//2, MARGIN + i*CELL,
                          text=str(i), font=("Consolas", 9), fill="#5A3E1B")

        # 화점 (천원/성)
        star_pts = [3, 5, 7] if BOARD_SIZE >= 9 else [BOARD_SIZE//2]
        for sr in star_pts:
            for sc in star_pts:
                x = MARGIN + sc * CELL
                y = MARGIN + sr * CELL
                c.create_oval(x-4, y-4, x+4, y+4, fill=COLOR_LINE, outline="")

        # 돌
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                v = self.game.board[row, col]
                if v == 0:
                    continue
                x = MARGIN + col * CELL
                y = MARGIN + row * CELL
                action = row * BOARD_SIZE + col
                is_last = (action == self.last_move)

                if v == 1:   # 흑
                    c.create_oval(x-STONE_R, y-STONE_R,
                                  x+STONE_R, y+STONE_R,
                                  fill=COLOR_BLACK, outline=COLOR_BLACK)
                    if is_last:
                        c.create_oval(x-7, y-7, x+7, y+7,
                                      fill=COLOR_LAST, outline="")
                else:        # 백
                    c.create_oval(x-STONE_R, y-STONE_R,
                                  x+STONE_R, y+STONE_R,
                                  fill=COLOR_WHITE, outline=COLOR_W_EDGE, width=1.5)
                    if is_last:
                        c.create_oval(x-7, y-7, x+7, y+7,
                                      fill=COLOR_LAST, outline="")

    def _set_status(self, text: str):
        self.status_var.set(text)

    # ── 실행 ──────────────────────────────────────────────────────
    def run(self):
        self.root.mainloop()


# ── 진입점 ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from neural_net import NeuralNetworkWrapper

    game = GomokuGame()
    net  = NeuralNetworkWrapper(game)

    best_path = os.path.join(CFG.model_directory, "best_model.pt")
    if os.path.exists(best_path):
        net.load_model("best_model")
        print("best_model.pt 로드 완료")
    else:
        print("저장된 모델 없음 - 랜덤 가중치로 실행합니다.")

    gui = GomokuGUI(net)
    gui.run()
