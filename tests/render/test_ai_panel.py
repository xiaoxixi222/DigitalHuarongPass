import pygame

from ai.agent import NeuralAgent
from ai.encoding import goal_state
from ai.model import HuarongNet
from render.ai_panel import draw_action_overlay, draw_ai_panel


def test_panel_draws_non_background_pixels():
    pygame.init()
    screen = pygame.Surface((1180, 720))
    trace = NeuralAgent(HuarongNet()).decide(goal_state(), step=0)

    draw_ai_panel(screen, pygame.Rect(800, 0, 380, 720), trace, paused=True)

    assert pygame.image.tostring(screen, "RGB") != bytes(1180 * 720 * 3)
    pygame.quit()


def test_action_overlay_draws_an_arrow():
    pygame.init()
    screen = pygame.Surface((800, 600))
    trace = NeuralAgent(HuarongNet()).decide(goal_state(), step=0)

    draw_action_overlay(screen, pygame.Rect(0, 0, 800, 600), 100, trace)

    assert pygame.image.tostring(screen, "RGB") != bytes(800 * 600 * 3)
    pygame.quit()
