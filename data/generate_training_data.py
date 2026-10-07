"""
InboxIQ - Synthetic Training Data Generator
=============================================
Generates a reproducible, fictional dataset of email records used ONLY to
train the category and priority ML models. This data is entirely synthetic
and does NOT contain any real Gmail messages.

Run:
    python data/generate_training_data.py

Output:
    data/training_emails.csv  (columns: email_id, subject, body, category, priority)
"""

import random
import os
import csv

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "training_emails.csv")

CATEGORIES = [
    "Education", "Work", "Finance", "Shopping", "Travel", "Social",
    "Security", "Promotions", "Personal", "Events", "Newsletter", "Other",
]

PRIORITIES = ["HIGH", "MEDIUM", "LOW"]

# Baseline urgency score (0-100) per category - reflects realistic patterns
# (security alerts tend to be urgent, newsletters rarely are). This is only
# a STARTING point: the final priority label for each email is computed from
# this baseline PLUS the actual urgency/deadline/security language that
# appears in that specific email's text (see compute_priority_label below),
# so the label genuinely depends on email content and is learnable by a
# text-based ML model - not just a fixed per-category lookup.
CATEGORY_BASE_SCORE = {
    "Security":   78,
    "Work":       52,
    "Finance":    50,
    "Travel":     48,
    "Education":  45,
    "Events":     40,
    "Personal":   35,
    "Other":      32,
    "Shopping":   26,
    "Social":     16,
    "Newsletter": 8,
    "Promotions": 6,
}

# Keyword groups used ONLY to compute the synthetic training label (this is
# separate from, and simpler than, the engineered features the real ML
# pipeline uses in src/preprocessing.py). Having the label depend on these
# words is what gives the priority model real signal to learn during
# training - it is not used at prediction time.
URGENCY_KEYWORDS = ["urgent", "immediately", "asap", "action required", "critical",
                     "alert", "important", "now", "hurry"]
DEADLINE_KEYWORDS = ["due", "deadline", "tomorrow", "today", "tonight", "within 24 hours",
                      "expires", "last date", "last chance", "ending soon", "ends tonight"]
SECURITY_KEYWORDS = ["password", "login", "verify", "security", "suspicious", "unauthorized",
                      "breach", "locked", "blocked", "verification code"]


def compute_priority_label(category, subject, body, rng):
    """
    Derive a HIGH/MEDIUM/LOW priority label from the category baseline plus
    the actual urgency/deadline/security language present in this email's
    text, with a small amount of random noise for realism. Thresholds were
    tuned so the overall label distribution stays reasonably balanced.
    """
    text = f"{subject} {body}".lower()
    score = CATEGORY_BASE_SCORE.get(category, 30)

    if any(kw in text for kw in URGENCY_KEYWORDS):
        score += 18
    if any(kw in text for kw in DEADLINE_KEYWORDS):
        score += 14
    if any(kw in text for kw in SECURITY_KEYWORDS):
        score += 16

    score += rng.uniform(-12, 12)  # small noise so the rule isn't perfectly deterministic
    score = max(0, min(100, score))

    if score >= 62:
        return "HIGH"
    elif score >= 35:
        return "MEDIUM"
    else:
        return "LOW"

NAMES = [
    "Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Ananya", "Karthik", "Divya",
    "Arjun", "Meera", "Rahul", "Kavya", "Siddharth", "Isha", "Nikhil", "Pooja",
    "Aditya", "Riya", "Varun", "Tanya", "Sarah", "James", "Emily", "Michael",
    "Olivia", "David", "Sophia", "Daniel",
]

COMPANIES = [
    "TechNova Solutions", "Brightline Corp", "Quantum Retail", "Skyline Travels",
    "Apex Bank", "Greenfield University", "Nimbus Cloud Services", "Vertex Systems",
    "BlueWave Airlines", "Everest Finance", "Orbit Electronics", "Harbor Insurance",
    "Crestline Academy", "Pulse Fitness", "Zenith Consulting", "Maple Street Cafe",
]

CITIES = [
    "Bangalore", "Hyderabad", "Mumbai", "Delhi", "Chennai", "Pune", "Goa",
    "Singapore", "Dubai", "London", "New York", "San Francisco",
]

PRODUCTS = [
    "wireless headphones", "running shoes", "laptop stand", "office chair",
    "smartphone case", "coffee maker", "backpack", "fitness tracker",
    "bluetooth speaker", "desk lamp", "water bottle", "yoga mat",
]

COURSES = [
    "Data Structures", "Machine Learning", "Operating Systems", "Database Systems",
    "Computer Networks", "Software Engineering", "Web Development", "Algorithms",
]


def rand_name():
    return random.choice(NAMES)


def rand_company():
    return random.choice(COMPANIES)


def rand_city():
    return random.choice(CITIES)


def rand_amount():
    return random.choice([499, 999, 1499, 2499, 4999, 9999, 14999, 24999])


def rand_date_phrase():
    return random.choice([
        "tomorrow", "on Monday", "this Friday", "in 3 days", "next week",
        "on 15th October", "by end of day", "within 24 hours", "on Saturday",
    ])


def build_education():
    name = rand_name()
    course = random.choice(COURSES)
    uni = random.choice(COMPANIES)
    templates = [
        (f"Your exam timetable has been released",
         f"Dear {name}, the semester examination timetable for {course} has been published on the student portal. "
         f"Please check your {uni} login for your seat number and reporting time. Attendance is mandatory."),
        (f"Assignment deadline reminder for {course}",
         f"Hi {name}, this is a reminder that your {course} assignment is due {rand_date_phrase()}. "
         f"Late submissions will attract a grade penalty. Please upload your file to the portal before the deadline."),
        (f"Result declared for {course}",
         f"Dear {name}, your semester result for {course} has been declared. Login to the student dashboard to view your grade sheet."),
        (f"Scholarship application open",
         f"Dear {name}, applications for the merit scholarship at {uni} are now open. Submit your documents before the last date to be considered."),
        (f"Class rescheduled",
         f"Hi {name}, the {course} lecture scheduled for today has been rescheduled to {rand_date_phrase()}. Please check the updated timetable."),
        (f"Library book return reminder",
         f"Dear {name}, the book you borrowed from the {uni} library is due for return {rand_date_phrase()}. A fine will apply after the due date."),
        (f"Semester fee payment due",
         f"Dear {name}, your semester fee of Rs.{rand_amount()} is due {rand_date_phrase()}. Please pay through the student portal to avoid late fees."),
        (f"Workshop registration confirmation",
         f"Hi {name}, you have successfully registered for the {course} workshop at {uni}. The session will be held {rand_date_phrase()}."),
    ]
    return random.choice(templates)


def build_work():
    name = rand_name()
    company = rand_company()
    templates = [
        (f"Your technical interview is scheduled",
         f"Dear {name}, your technical interview with {company} is scheduled for {rand_date_phrase()}. "
         f"Please be available on video call 10 minutes before the scheduled time and carry your resume."),
        (f"Offer letter attached",
         f"Dear {name}, congratulations! Please find attached your offer letter from {company}. Kindly confirm your acceptance {rand_date_phrase()}."),
        (f"Project status meeting",
         f"Hi {name}, we have a project status review meeting {rand_date_phrase()}. Please prepare your module updates beforehand."),
        (f"Urgent: Production issue needs attention",
         f"Hi {name}, we are seeing a critical issue in production at {company}. Please join the incident call immediately to help resolve it."),
        (f"Performance review scheduled",
         f"Dear {name}, your annual performance review at {company} has been scheduled {rand_date_phrase()}. Please complete your self-assessment form before the meeting."),
        (f"Team lunch this Friday",
         f"Hi {name}, join us for the team lunch this Friday at {rand_city()}. Please confirm your attendance."),
        (f"Leave request approved",
         f"Dear {name}, your leave request has been approved by your manager at {company}. Please plan your handover accordingly."),
        (f"Client escalation - action required",
         f"Hi {name}, a client at {company} has escalated a concern and needs a response {rand_date_phrase()}. Please review and reply at the earliest."),
    ]
    return random.choice(templates)


def build_finance():
    name = rand_name()
    bank = random.choice(["Apex Bank", "Everest Finance", "Harbor Insurance", "National Credit Union"])
    amount = rand_amount()
    templates = [
        (f"Your credit card payment is due",
         f"Dear {name}, your credit card bill of Rs.{amount} with {bank} is due {rand_date_phrase()}. Pay now to avoid late payment charges and interest."),
        (f"Loan EMI deduction notice",
         f"Dear {name}, your EMI of Rs.{amount} will be deducted from your account {rand_date_phrase()}. Please ensure sufficient balance."),
        (f"Your account statement is ready",
         f"Hi {name}, your monthly account statement from {bank} is now available for download on the net banking portal."),
        (f"Insurance premium due reminder",
         f"Dear {name}, your insurance premium of Rs.{amount} is due {rand_date_phrase()}. Renew now to keep your policy active."),
        (f"Tax filing deadline approaching",
         f"Dear {name}, the deadline to file your income tax return is {rand_date_phrase()}. Please complete your filing to avoid penalties."),
        (f"Salary credited to your account",
         f"Dear {name}, your salary of Rs.{amount*3} has been credited to your {bank} account."),
        (f"Low balance alert",
         f"Dear {name}, your account balance at {bank} is below the minimum required limit. Please deposit funds to avoid charges."),
        (f"Fixed deposit maturity notice",
         f"Dear {name}, your fixed deposit with {bank} is maturing {rand_date_phrase()}. Visit the branch or portal to renew or withdraw."),
    ]
    return random.choice(templates)


def build_shopping():
    name = rand_name()
    product = random.choice(PRODUCTS)
    order_id = random.randint(100000, 999999)
    templates = [
        (f"Your order has been shipped",
         f"Hi {name}, great news! Your order #{order_id} for {product} has been shipped and will arrive {rand_date_phrase()}. Track your package online."),
        (f"Order delivered successfully",
         f"Hi {name}, your order #{order_id} for {product} has been delivered. We hope you enjoy your purchase!"),
        (f"Your order is out for delivery",
         f"Hi {name}, your package containing {product} is out for delivery and should reach you today."),
        (f"Return confirmation for your order",
         f"Hi {name}, we have received your return request for {product} from order #{order_id}. Refund will be processed within 5-7 business days."),
        (f"Price drop alert on your wishlist item",
         f"Hi {name}, the {product} in your wishlist just dropped in price. Grab it before the offer ends!"),
        (f"Payment received for order #{order_id}",
         f"Hi {name}, we have received your payment of Rs.{rand_amount()} for {product}. Your order is being processed."),
        (f"Your cart items are waiting",
         f"Hi {name}, you left {product} in your cart. Complete your purchase before it goes out of stock."),
        (f"Invoice for your recent purchase",
         f"Hi {name}, please find the invoice attached for your purchase of {product}, order #{order_id}."),
    ]
    return random.choice(templates)


def build_travel():
    name = rand_name()
    city = rand_city()
    airline = random.choice(["BlueWave Airlines", "Skyline Travels", "Orbit Airways"])
    templates = [
        (f"Your flight departs tomorrow",
         f"Dear {name}, this is a reminder that your {airline} flight to {city} departs {rand_date_phrase()}. Please check in online and reach the airport 2 hours early."),
        (f"Booking confirmation - {city} trip",
         f"Dear {name}, your trip to {city} has been confirmed with {airline}. Your e-ticket and itinerary are attached."),
        (f"Flight schedule change notice",
         f"Dear {name}, your {airline} flight to {city} has been rescheduled. Please review the new departure time {rand_date_phrase()}."),
        (f"Hotel booking confirmed in {city}",
         f"Dear {name}, your hotel reservation in {city} is confirmed for check-in {rand_date_phrase()}. Present this confirmation at the front desk."),
        (f"Check-in now open for your flight",
         f"Dear {name}, online check-in for your {airline} flight is now open. Check in now to select your preferred seat."),
        (f"Your visa application status updated",
         f"Dear {name}, your visa application for travel to {city} has been updated. Please log in to the portal to view details."),
        (f"Travel itinerary for your trip",
         f"Dear {name}, here is your complete travel itinerary for the {city} trip, including flights, hotel and local transport."),
        (f"Gate change for your flight",
         f"Dear {name}, the departure gate for your {airline} flight to {city} has changed. Please proceed to the new gate shown on the display."),
    ]
    return random.choice(templates)


def build_social():
    name = rand_name()
    friend = rand_name()
    templates = [
        (f"{friend} mentioned you in a comment",
         f"Hi {name}, {friend} mentioned you in a comment on their recent post. Tap to view and reply."),
        (f"{friend} sent you a friend request",
         f"Hi {name}, {friend} wants to connect with you. Accept the request to start chatting."),
        (f"You have a new follower",
         f"Hi {name}, {friend} started following your profile. Check out their updates too."),
        (f"{friend} liked your photo",
         f"Hi {name}, {friend} and 12 others liked the photo you posted recently."),
        (f"New message from {friend}",
         f"Hi {name}, you have a new direct message from {friend}. Open the app to read it."),
        (f"{friend} tagged you in a post",
         f"Hi {name}, {friend} tagged you in a post. See what they shared with you."),
        (f"Weekly activity summary",
         f"Hi {name}, here is a summary of your social activity this week, including new followers and comments."),
        (f"{friend} shared a memory with you",
         f"Hi {name}, {friend} shared a memory from a past event. Take a look back at this moment."),
    ]
    return random.choice(templates)


def build_security():
    name = rand_name()
    templates = [
        (f"New login detected on your account",
         f"Dear {name}, we detected a new login to your account from an unrecognized device in {rand_city()}. "
         f"If this wasn't you, please reset your password immediately and review your account activity."),
        (f"Your password was changed",
         f"Dear {name}, your account password was recently changed. If you did not make this change, contact support immediately and secure your account."),
        (f"Suspicious activity detected",
         f"Dear {name}, we noticed unusual sign-in attempts on your account. Please verify your identity now to prevent unauthorized access."),
        (f"Two-factor authentication code",
         f"Dear {name}, your one-time verification code is required to complete sign-in. This code will expire in 10 minutes. Do not share it with anyone."),
        (f"Security alert: Action required",
         f"Dear {name}, your account security settings need urgent attention. Please verify your recovery email and phone number within 24 hours."),
        (f"Unusual sign-in attempt blocked",
         f"Dear {name}, we blocked a sign-in attempt from a new location. Please confirm whether this was you to keep your account secure."),
        (f"Your account has been temporarily locked",
         f"Dear {name}, we locked your account after multiple failed login attempts. Verify your identity immediately to restore access."),
        (f"Data breach notification",
         f"Dear {name}, a third-party service you use reported a data breach. We strongly recommend changing your password right away as a precaution."),
    ]
    return random.choice(templates)


def build_promotions():
    product = random.choice(PRODUCTS)
    discount = random.choice([10, 20, 30, 40, 50, 60, 70])
    templates = [
        (f"{discount}% discount available this weekend",
         f"Don't miss out! Get {discount}% off on {product} this weekend only. Shop now before the offer expires."),
        (f"Flash sale: Up to {discount}% off",
         f"Hurry! Our flash sale offers up to {discount}% off on {product} and more. Limited stock available."),
        (f"Exclusive deal just for you",
         f"As a valued customer, enjoy an exclusive {discount}% discount on your next purchase of {product}."),
        (f"Buy one get one free offer",
         f"For a limited time, buy one {product} and get another absolutely free. Offer valid while stocks last."),
        (f"Mega sale starts now",
         f"Our biggest sale of the season is live! Save up to {discount}% on {product} and thousands of other items."),
        (f"Last chance to save {discount}%",
         f"This is your last chance to save {discount}% on {product}. The offer ends tonight at midnight."),
        (f"New arrivals with special pricing",
         f"Check out our new collection of {product} with special launch pricing, available for a limited time."),
        (f"Clearance sale - everything must go",
         f"Clearance sale is on! Get massive discounts on {product} and other items before stock runs out."),
    ]
    return random.choice(templates)


def build_personal():
    name = rand_name()
    relative = random.choice(["mom", "dad", "sister", "brother", "cousin", "grandma", "uncle"])
    templates = [
        (f"Meeting with family this Sunday",
         f"Hey {name}, just a reminder that we are all meeting at {relative}'s place this Sunday for lunch. Let me know if you can make it."),
        (f"Happy birthday!",
         f"Hey {name}, wishing you a very happy birthday! Hope you have a wonderful day filled with joy."),
        (f"Can you call me when free?",
         f"Hi {name}, give me a call whenever you're free, nothing urgent, just wanted to catch up."),
        (f"Photos from last weekend",
         f"Hey {name}, sharing the photos from our trip last weekend. Let me know which ones you want printed."),
        (f"Dinner plans for this week",
         f"Hi {name}, are you free for dinner {rand_date_phrase()}? Let's catch up over food."),
        (f"Reminder about {relative}'s visit",
         f"Hey {name}, just reminding you that {relative} is visiting us {rand_date_phrase()}. Let's plan accordingly."),
        (f"Congratulations on your achievement",
         f"Hi {name}, heard about your recent achievement, huge congratulations! So proud of you."),
        (f"Need your help with something",
         f"Hey {name}, could you help me with something small this weekend? Let me know when you're around."),
    ]
    return random.choice(templates)


def build_events():
    name = rand_name()
    event = random.choice(["Tech Conference 2026", "Annual Alumni Meet", "Music Festival", "Startup Summit", "Developer Meetup", "Cultural Fest"])
    city = rand_city()
    templates = [
        (f"Conference registration is open",
         f"Dear {name}, registrations for {event} in {city} are now open. Register early to get the best seats and discounted pricing."),
        (f"Your event ticket is confirmed",
         f"Dear {name}, your ticket for {event} has been confirmed. The event will be held in {city} {rand_date_phrase()}."),
        (f"Reminder: Event starts soon",
         f"Dear {name}, {event} starts {rand_date_phrase()} in {city}. Don't forget to bring your ticket and ID proof."),
        (f"Speaker lineup announced",
         f"Dear {name}, the speaker lineup for {event} has been announced. Check out the full schedule on our website."),
        (f"Invitation to {event}",
         f"Dear {name}, you are cordially invited to attend {event} in {city}. We would love to see you there."),
        (f"Event venue changed",
         f"Dear {name}, please note that the venue for {event} has changed. Check the updated location before you travel."),
        (f"Early bird offer ending soon",
         f"Dear {name}, the early bird discount for {event} ends {rand_date_phrase()}. Register now to save on your ticket."),
        (f"Thank you for attending {event}",
         f"Dear {name}, thank you for attending {event}. We hope you enjoyed the sessions and networking opportunities."),
    ]
    return random.choice(templates)


def build_newsletter():
    topic = random.choice(["technology", "finance", "health", "travel", "sports", "design", "startups", "science"])
    templates = [
        (f"Weekly {topic} newsletter",
         f"Here is your weekly roundup of the top {topic} stories, curated just for you. Read on for this week's highlights."),
        (f"This week in {topic}",
         f"Catch up on everything that happened in the world of {topic} this week with our curated digest."),
        (f"Monthly {topic} digest",
         f"Your monthly digest of the most important {topic} updates, articles, and insights is here."),
        (f"Top 5 {topic} stories this week",
         f"We picked the top 5 stories in {topic} this week so you don't have to scroll through everything yourself."),
        (f"{topic.capitalize()} trends you should know",
         f"Explore the latest trends in {topic} with insights from industry experts in this edition of our newsletter."),
        (f"Editor's picks: {topic} edition",
         f"Our editors have handpicked the best {topic} reads of the week for your newsletter digest."),
        (f"Your {topic} briefing",
         f"A short and useful briefing on {topic} news, delivered straight to your inbox as always."),
        (f"Subscribe update: {topic} newsletter",
         f"Thanks for subscribing! Here's the latest edition of our {topic} newsletter packed with updates."),
    ]
    return random.choice(templates)


def build_other():
    name = rand_name()
    templates = [
        (f"Please confirm your email address",
         f"Hi {name}, please click the link below to confirm your email address and complete your account setup."),
        (f"Feedback requested",
         f"Hi {name}, we would love to hear your feedback about your recent experience with us. It only takes a minute."),
        (f"Service update notice",
         f"Dear {name}, we are making some updates to our service. There may be brief downtime {rand_date_phrase()}."),
        (f"Survey: Tell us what you think",
         f"Hi {name}, help us improve by taking this short survey about your recent experience."),
        (f"Terms of service update",
         f"Dear {name}, we have updated our terms of service. Please review the changes at your convenience."),
        (f"Welcome aboard!",
         f"Hi {name}, welcome! We're glad to have you with us. Explore your account to get started."),
        (f"Reminder: Complete your profile",
         f"Hi {name}, your profile is incomplete. Add a few more details to get the most out of our platform."),
        (f"Miscellaneous notification",
         f"Hi {name}, this is a general notification regarding your recent activity with us."),
    ]
    return random.choice(templates)


CATEGORY_BUILDERS = {
    "Education": build_education,
    "Work": build_work,
    "Finance": build_finance,
    "Shopping": build_shopping,
    "Travel": build_travel,
    "Social": build_social,
    "Security": build_security,
    "Promotions": build_promotions,
    "Personal": build_personal,
    "Events": build_events,
    "Newsletter": build_newsletter,
    "Other": build_other,
}

EMAILS_PER_CATEGORY = 150  # 12 categories * 150 = 1800 total records


def generate_dataset():
    rows = []
    email_id = 1
    for category in CATEGORIES:
        builder = CATEGORY_BUILDERS[category]
        seen_subjects = set()
        attempts = 0
        count = 0
        while count < EMAILS_PER_CATEGORY and attempts < EMAILS_PER_CATEGORY * 10:
            attempts += 1
            subject, body = builder()
            key = (subject, body)
            # allow repeats since templates are randomized with names/amounts,
            # but skip exact duplicate (subject, body) pairs when possible
            if key in seen_subjects and attempts < EMAILS_PER_CATEGORY * 8:
                continue
            seen_subjects.add(key)
            priority = compute_priority_label(category, subject, body, random)
            rows.append({
                "email_id": email_id,
                "subject": subject,
                "body": body,
                "category": category,
                "priority": priority,
            })
            email_id += 1
            count += 1
    random.shuffle(rows)
    # re-number email_id after shuffle for clean sequential ids
    for idx, row in enumerate(rows, start=1):
        row["email_id"] = idx
    return rows


def main():
    rows = generate_dataset()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["email_id", "subject", "body", "category", "priority"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} synthetic training emails.")
    print(f"Saved to: {OUTPUT_PATH}")

    # quick balance summary
    from collections import Counter
    cat_counts = Counter(r["category"] for r in rows)
    pri_counts = Counter(r["priority"] for r in rows)
    print("\nCategory distribution:")
    for c in CATEGORIES:
        print(f"  {c:12s}: {cat_counts[c]}")
    print("\nPriority distribution:")
    for p in PRIORITIES:
        print(f"  {p:8s}: {pri_counts[p]}")


if __name__ == "__main__":
    main()
