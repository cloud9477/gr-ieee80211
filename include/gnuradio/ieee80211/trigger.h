/* -*- c++ -*- */
/*
 * Copyright 2022 Zelin Yun.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_IEEE80211_TRIGGER_H
#define INCLUDED_IEEE80211_TRIGGER_H

#include <gnuradio/ieee80211/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace ieee80211 {

    /*!
     * \brief <+description of block+>
     * \ingroup ieee80211
     *
     */
    class IEEE80211_API trigger : virtual public gr::block
    {
     public:
      typedef std::shared_ptr<trigger> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of ieee80211::trigger.
       *
       * To avoid accidental use of raw pointers, ieee80211::trigger's
       * constructor is in a private implementation
       * class. ieee80211::trigger::make is the public interface for
       * creating new instances.
       */
      static sptr make();
    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_TRIGGER_H */
