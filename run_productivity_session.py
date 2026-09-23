#!/usr/bin/env python3
"""
Run a design thinking session for the productivity app concept using ReqGPT's mentor pipeline.
"""

from mentor import process_mentor_turn, build_mentor_session, print_ascii_dashboard, MentorSession

def run_session():
    project_name = "ProductivityApp"
    username = "User"
    model_name = "optimized-pods"

    print("=" * 60)
    print("DESIGN THINKING SESSION: Productivity App Concept")
    print("=" * 60)
    print()

    # Build session
    session = build_mentor_session(project_name)
    
    # The initial concept - feed it as the first user message
    initial_concept = """I want to explore a productivity app concept. Here's the core idea:

**Problem**: Many productivity apps exist (Habitica, Finch, Forest, Todoist, TickTick, Motion, Sunsama, etc.) but people still struggle with productivity. The issue isn't planning - it's STARTING. People know what to do (study, gym, work) but don't begin.

**Core Mission**: Build an app that reduces the friction of starting important work - an adaptive productivity coach that learns from user behavior.

**Key Features**:
1. AI suggests task breakdown (not manual subtask creation)
2. Reminders offer choices: Start / Delay / Can't Start
3. If user delays/repeats "Can't Start", app asks "What stopped you?" (tired, overwhelmed, distracted, busy, don't feel like it)
4. App adapts response based on reason (e.g., overwhelmed → "Just open your notes"; tired → "Read one page")
5. Learns patterns: when user completes coding tasks, skips workouts, delays but then continues
6. Privacy-first, offline-capable
7. No gamification/pet/XP/leaderboards - every feature must answer: "Does this help the user START important work?"

**Success Metric**: How often users actually START planned work after reminder (avg start delay, % starting within 5-10 min, reduction in delay over time)

Please help me explore this through design thinking questioning."""

    print("User: [Initial concept provided]")
    print("-" * 60)
    
    reply, session, _, _ = process_mentor_turn(
        user_message=initial_concept,
        username=username,
        project_name=project_name,
        model_name=model_name
    )
    
    print(f"Coach: {reply}")
    print()
    print_ascii_dashboard(session)
    
    # Continue with follow-up exploration
    followups = [
        "The core insight is that STARTING is the bottleneck, not planning. How might we validate this is truly the universal problem vs. just my assumption?",
        "What assumptions am I making about user behavior that could be wrong? For example, assuming people WANT to start but can't vs. they don't actually want to do the work.",
        "How might we test the adaptive coaching approach without building the full app first?",
        "What would make this different from existing 'smart reminder' features in apps like Todoist or Google Calendar?",
        "The app learns patterns locally. What privacy concerns might users have even with local-only data?",
        "How do we measure 'starting' reliably without being intrusive?",
        "What happens when the app's suggestion is wrong? How does the system recover trust?",
        "Is there a risk that adaptive coaching becomes another form of nagging?",
    ]
    
    for i, followup in enumerate(followups, 1):
        print(f"\n{'='*60}")
        print(f"Follow-up {i}: {followup}")
        print(f"{'='*60}")
        
        reply, session, _, _ = process_mentor_turn(
            user_message=followup,
            username=username,
            project_name=project_name,
            model_name=model_name
        )
        
        print(f"Coach: {reply}")
        print()
        print_ascii_dashboard(session)

    print("\n" + "=" * 60)
    print("SESSION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_session()